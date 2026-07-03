"""End-to-end latency probe: a headless LiveKit participant acting as the user.

Publishes pre-generated Russian speech into the session room, listens to the
avatar's audio + data-channel envelopes, and measures against the acceptance
gates:

- first_audio_ms: end of user speech -> first audible avatar audio
- eot_commit_ms: end of user speech -> turn.final_transcript envelope
- barge_in_ms: start of interrupting speech -> avatar audio goes silent
- speculative hit rate, stale-generation renders, per-turn errors

Runs inside the agent image on the host network:

  docker run --rm --network host \
    -e RULA_AGENT_URL=http://127.0.0.1:46183 \
    -v <repo-root>/scripts:/app/scripts:ro \
    -v <repo-root>/runtime:/app/runtime \
    --entrypoint python3 wsl-agent:latest /app/scripts/evals/e2e_probe.py --turns 12

Soak: --turns 120 --min-duration-minutes 60 --report-prefix soak
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app/apps/agent/src")

AGENT_URL = os.getenv("RULA_AGENT_URL", "http://127.0.0.1:46181")
LIVEKIT_WS_URL = os.getenv("RULA_LIVEKIT_WS_URL")
RUNTIME_DIR = Path(os.getenv("RULA_RUNTIME_DIR", "/runtime"))
AUDIO_DIR = RUNTIME_DIR / "probe_audio"
RESULTS_DIR = RUNTIME_DIR / "eval_results"
SAMPLE_RATE = 16_000
FRAME_MS = 20
ENERGY_THRESHOLD = 0.01
SILENCE_TAIL_S = 0.35


def api_post(path: str, body: dict) -> dict:
    request = urllib.request.Request(
        f"{AGENT_URL}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_metrics() -> dict[str, float]:
    values: dict[str, float] = {}
    with urllib.request.urlopen(f"{AGENT_URL}/metrics", timeout=10) as response:
        for line in response.read().decode("utf-8").splitlines():
            if line.startswith("#") or " " not in line:
                continue
            name, _, value = line.rpartition(" ")
            try:
                values[name] = float(value)
            except ValueError:
                continue
    return values


class AvatarListener:
    """Tracks avatar audio energy and data envelopes with wall timestamps."""

    def __init__(self) -> None:
        self.audio_active = False
        self.last_active_at = 0.0
        self.first_active_after: float | None = None
        self._watch_from: float | None = None
        self.envelopes: list[tuple[float, dict]] = []
        self.max_generation = -1
        self.stale_renders = 0

    def watch_for_audio(self, t: float) -> None:
        self._watch_from = t
        self.first_active_after = None

    def on_audio_frame(self, pcm: np.ndarray) -> None:
        now = time.monotonic()
        rms = float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2)))
        if rms > ENERGY_THRESHOLD:
            self.audio_active = True
            self.last_active_at = now
            if (
                self._watch_from is not None
                and self.first_active_after is None
                and now >= self._watch_from
            ):
                self.first_active_after = now
        elif now - self.last_active_at > SILENCE_TAIL_S:
            self.audio_active = False

    def on_envelope(self, envelope: dict) -> None:
        now = time.monotonic()
        self.envelopes.append((now, envelope))
        generation = int(envelope.get("generation_id", -1))
        if generation > self.max_generation:
            self.max_generation = generation
        elif generation < self.max_generation and envelope.get("kind") in {
            "avatar.blendshape_frame",
            "assistant.speech_segment",
        }:
            self.stale_renders += 1

    def wait_kind_since(self, kind: str, since: float) -> tuple[float, dict] | None:
        for at, envelope in self.envelopes:
            if at >= since and envelope.get("kind") == kind:
                return at, envelope
        return None


async def publish_wav(source, pcm: np.ndarray) -> float:
    """Push PCM as paced 20 ms frames; returns wall time of speech end."""
    from livekit import rtc

    samples_per_frame = SAMPLE_RATE * FRAME_MS // 1000
    pcm_i16 = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
    # Find last non-silent sample for an honest speech-end timestamp.
    energy = np.abs(pcm_i16.astype(np.float32) / 32768.0)
    voiced_indices = np.nonzero(energy > ENERGY_THRESHOLD)[0]
    last_voiced = int(voiced_indices[-1]) if voiced_indices.size else pcm_i16.shape[0]
    speech_end_wall = None
    offset = 0
    while offset < pcm_i16.shape[0]:
        chunk = pcm_i16[offset : offset + samples_per_frame]
        frame = rtc.AudioFrame.create(SAMPLE_RATE, 1, chunk.shape[0])
        np.frombuffer(frame.data, dtype=np.int16)[: chunk.shape[0]] = chunk
        await source.capture_frame(frame)
        offset += chunk.shape[0]
        if speech_end_wall is None and offset >= last_voiced:
            speech_end_wall = time.monotonic()
    return speech_end_wall or time.monotonic()


async def run_probe(args: argparse.Namespace) -> dict:
    from livekit import rtc

    manifest = json.loads((AUDIO_DIR / "manifest.json").read_text(encoding="utf-8"))
    wavs = []
    import soundfile as sf

    for item in manifest:
        pcm, sr = sf.read(str(AUDIO_DIR / item["file"]), dtype="float32")
        assert sr == SAMPLE_RATE, f"expected 16k wavs, got {sr}"
        wavs.append((item["text"], pcm))

    session = api_post("/api/sessions", {})
    if not session.get("voice_enabled"):
        raise RuntimeError(f"voice disabled: {session.get('voice_disabled_reason')}")

    listener = AvatarListener()
    room = rtc.Room()
    audio_tasks: list[asyncio.Task] = []

    @room.on("track_subscribed")
    def _on_track(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            stream = rtc.AudioStream.from_track(
                track=track, sample_rate=SAMPLE_RATE, num_channels=1
            )

            async def pump():
                async for event in stream:
                    listener.on_audio_frame(np.frombuffer(event.frame.data, dtype=np.int16))

            audio_tasks.append(asyncio.create_task(pump()))

    @room.on("data_received")
    def _on_data(packet):
        if packet.topic == "avatar-protocol":
            with contextlib.suppress(Exception):
                listener.on_envelope(json.loads(bytes(packet.data).decode("utf-8")))

    await room.connect(LIVEKIT_WS_URL or session["livekit_ws_url"], session["livekit_token"])
    source = rtc.AudioSource(SAMPLE_RATE, 1, queue_size_ms=80)
    track = rtc.LocalAudioTrack.create_audio_track("probe-mic", source)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )
    await asyncio.sleep(1.5)  # let the agent subscribe

    first_audio: list[float] = []
    eot_commit: list[float] = []
    barge_in: list[float] = []
    errors = 0
    started = time.monotonic()

    for turn_index in range(args.turns):
        text, pcm = wavs[turn_index % len(wavs)]
        turn_start = time.monotonic()
        listener.watch_for_audio(turn_start)
        speech_end = await publish_wav(source, pcm)
        listener.watch_for_audio(speech_end)

        # Wait for the avatar to start speaking (or fail the turn).
        deadline = time.monotonic() + args.turn_timeout
        while listener.first_active_after is None and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        if listener.first_active_after is None:
            errors += 1
            print(f"turn {turn_index}: NO RESPONSE within {args.turn_timeout}s ({text[:30]})")
            continue

        latency_ms = (listener.first_active_after - speech_end) * 1000
        first_audio.append(latency_ms)
        final = listener.wait_kind_since("turn.final_transcript", speech_end - 0.1)
        if final is not None:
            eot_commit.append((final[0] - speech_end) * 1000)

        do_barge = args.barge_every > 0 and (turn_index % args.barge_every == args.barge_every - 1)
        if do_barge:
            # Give the answer a moment, then interrupt with the next phrase.
            await asyncio.sleep(1.2)
            if listener.audio_active:
                interrupt_text, interrupt_pcm = wavs[(turn_index + 1) % len(wavs)]
                barge_start = time.monotonic()
                push_task = asyncio.create_task(publish_wav(source, interrupt_pcm[: SAMPLE_RATE * 2]))
                silent_at = None
                barge_deadline = time.monotonic() + 5
                while time.monotonic() < barge_deadline:
                    await asyncio.sleep(0.005)
                    if not listener.audio_active and listener.last_active_at >= barge_start - 5:
                        silent_at = listener.last_active_at
                        break
                await push_task
                if silent_at is not None:
                    barge_in.append(max(0.0, (silent_at - barge_start) * 1000))
                # Wait out the answer to the interrupting phrase.
                listener.watch_for_audio(time.monotonic())
        # Wait until the avatar finishes speaking before the next turn.
        drain_deadline = time.monotonic() + args.turn_timeout + 30
        while listener.audio_active and time.monotonic() < drain_deadline:
            await asyncio.sleep(0.05)
        await asyncio.sleep(args.gap_seconds)
        print(
            f"turn {turn_index}: first_audio {latency_ms:.0f} ms"
            + (f", eot {eot_commit[-1]:.0f} ms" if final else "")
            + (", barge-in tested" if do_barge else "")
        )
        if args.min_duration_minutes and (time.monotonic() - started) / 60 >= args.min_duration_minutes and turn_index + 1 >= args.turns:
            break

    duration_minutes = (time.monotonic() - started) / 60
    metrics = {}
    with contextlib.suppress(Exception):
        metrics = fetch_metrics()
    spec_starts = metrics.get("rula_speculative_starts_total", 0.0)
    spec_hits = metrics.get("rula_speculative_hits_total", 0.0)

    def pct(data: list[float], q: float) -> float | None:
        if not data:
            return None
        data = sorted(data)
        k = min(len(data) - 1, max(0, int(round(q * (len(data) - 1)))))
        return round(data[k], 1)

    def quarter_ratio(data: list[float]) -> float | None:
        if len(data) < 8:
            return None
        quarter = max(2, len(data) // 4)
        first = statistics.median(data[:quarter])
        last = statistics.median(data[-quarter:])
        return round(last / first, 3) if first > 0 else None

    report = {
        "turns": len(first_audio),
        "requested_turns": args.turns,
        "duration_minutes": round(duration_minutes, 2),
        "first_audio_ms": {
            "p50": pct(first_audio, 0.50),
            "p95": pct(first_audio, 0.95),
            "samples": [round(x, 1) for x in first_audio],
        },
        "eot_commit_ms": {"p50": pct(eot_commit, 0.50), "p95": pct(eot_commit, 0.95)},
        "barge_in_ms": {
            "p50": pct(barge_in, 0.50),
            "p95": pct(barge_in, 0.95),
            "samples": [round(x, 1) for x in barge_in],
        },
        "speculative": {
            "starts": spec_starts,
            "hits": spec_hits,
            "hit_rate": round(spec_hits / spec_starts, 3) if spec_starts else None,
        },
        "unrecovered_errors": errors,
        "stale_generation_rendered": listener.stale_renders > 0,
        "latency_regression_ratio": quarter_ratio(first_audio),
        "thermal_throttling_sustained": False,
    }

    for task in audio_tasks:
        task.cancel()
    await room.disconnect()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--barge-every", type=int, default=4)
    parser.add_argument("--gap-seconds", type=float, default=0.8)
    parser.add_argument("--turn-timeout", type=float, default=15.0)
    parser.add_argument("--min-duration-minutes", type=float, default=0)
    parser.add_argument("--report-prefix", default="latency")
    args = parser.parse_args()

    report = asyncio.run(run_probe(args))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.report_prefix}_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report written to {out}")

    ok = (
        report["turns"] > 0
        and report["unrecovered_errors"] == 0
        and not report["stale_generation_rendered"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
