"""GPU smoke test for the in-process voice engines (runs inside the agent container).

Round-trip: Qwen3-TTS synthesizes Russian speech -> GigaAM transcribes it back
-> Audio2Face-3D animates it. Prints wall times and sanity checks; exits
non-zero on failure so it can gate releases.

Usage (from repo root on Windows):
  docker compose -f infra/wsl/docker-compose.yml run --rm --entrypoint python3 agent /app/scripts/evals/voice_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app/apps/agent/src")

MODELS = Path("/app/models/hf")
PHRASES = [
    "Привет! Расскажи, пожалуйста, какая сегодня погода в Москве?",
    "Мне нравится разговаривать с тобой по-русски.",
]


def normalize(text: str) -> str:
    import re

    return re.sub(r"[\W_]+", "", text.lower())


def main() -> int:
    failures: list[str] = []

    print("=== TTS: Qwen3-TTS-12Hz-1.7B-CustomVoice ===", flush=True)
    from ru_local_avatar_agent.voice.tts import QwenTtsEngine

    t0 = time.monotonic()
    tts = QwenTtsEngine(MODELS / "Qwen__Qwen3-TTS-12Hz-1.7B-CustomVoice")
    print(f"loaded in {time.monotonic() - t0:.1f}s", flush=True)

    clauses = []
    for phrase in PHRASES:
        t0 = time.monotonic()
        clause = tts.synthesize_blocking(phrase)
        dt = time.monotonic() - t0
        duration = clause.pcm.shape[0] / clause.sample_rate
        rms = float(np.sqrt(np.mean(clause.pcm**2)))
        print(
            f"synth {dt*1000:.0f} ms for {duration:.2f}s audio @{clause.sample_rate}Hz "
            f"(RTF {dt/duration:.2f}, rms {rms:.3f}): {phrase[:40]}...",
            flush=True,
        )
        clauses.append(clause)
        if rms < 0.01:
            failures.append(f"TTS produced near-silent audio for: {phrase}")

    print("=== STT: GigaAM-v3 e2e_rnnt round-trip ===", flush=True)
    from ru_local_avatar_agent.voice.audio import resample
    from ru_local_avatar_agent.voice.stt import GigaAMTranscriber

    t0 = time.monotonic()
    stt = GigaAMTranscriber(MODELS / "ai-sage__GigaAM-v3")
    print(f"loaded in {time.monotonic() - t0:.1f}s", flush=True)

    for phrase, clause in zip(PHRASES, clauses):
        pcm16 = resample(clause.pcm, clause.sample_rate, 16_000)
        t0 = time.monotonic()
        text = stt.transcribe(pcm16)
        dt = time.monotonic() - t0
        print(f"stt {dt*1000:.0f} ms: {text!r}", flush=True)
        ref, hyp = normalize(phrase), normalize(text)
        overlap = sum(1 for a, b in zip(ref, hyp) if a == b) / max(len(ref), 1)
        if not hyp:
            failures.append(f"STT returned empty text for: {phrase}")
        elif overlap < 0.6 and hyp not in ref and ref not in hyp:
            print(f"  WARNING: weak round-trip match ({overlap:.2f}) ref={ref[:40]} hyp={hyp[:40]}")

        # Partial-latency check on a short prefix (the hot path runs this every ~280 ms).
        prefix = pcm16[: 16_000 * 2]
        t0 = time.monotonic()
        stt.transcribe(prefix)
        print(f"stt partial (2s audio) {1000*(time.monotonic()-t0):.0f} ms", flush=True)

    print("=== A2F: Audio2Face-3D v3.0 ===", flush=True)
    from ru_local_avatar_agent.voice.face import A2FEngine, A2FTurnStream

    t0 = time.monotonic()
    engine = A2FEngine(MODELS / "nvidia__Audio2Face-3D-v3.0")
    print(f"loaded in {time.monotonic() - t0:.1f}s, provider={engine.provider}", flush=True)
    if engine.provider != "CUDAExecutionProvider":
        failures.append(f"A2F not on CUDA: {engine.provider}")

    stream = A2FTurnStream(engine)
    clause = clauses[0]
    pcm16 = resample(clause.pcm, clause.sample_rate, 16_000)
    t0 = time.monotonic()
    frames = stream.push_audio(pcm16)
    frames += stream.flush()
    dt = time.monotonic() - t0
    duration = pcm16.shape[0] / 16_000
    jaw = [f.values.get("jawOpen", 0.0) for f in frames]
    print(
        f"a2f {dt*1000:.0f} ms for {duration:.2f}s audio -> {len(frames)} frames @30fps "
        f"(jawOpen min {min(jaw):.3f} max {max(jaw):.3f} mean {np.mean(jaw):.3f})",
        flush=True,
    )
    expected_frames = int(duration * 30)
    if len(frames) < expected_frames - 15:
        failures.append(f"A2F produced too few frames: {len(frames)} < ~{expected_frames}")
    if max(jaw) < 0.05:
        failures.append("A2F jawOpen never rises above 0.05 — lipsync looks dead")
    if max(jaw) > 0.999:
        print("  WARNING: jawOpen saturates at 1.0 — check delta/absolute detection")

    print("=== VAD: Silero ===", flush=True)
    from ru_local_avatar_agent.voice.vad import SileroVad

    vad = SileroVad()
    speech_probs = [
        vad.probability(pcm16[i : i + 512]) for i in range(0, 512 * 40, 512)
    ]
    vad.reset()
    silence_probs = [
        vad.probability(np.zeros(512, dtype=np.float32)) for _ in range(20)
    ]
    print(
        f"vad speech mean {np.mean(speech_probs):.2f}, silence mean {np.mean(silence_probs):.2f}",
        flush=True,
    )
    if np.mean(speech_probs) < 0.5:
        failures.append("VAD does not detect TTS speech as voiced")
    if np.mean(silence_probs) > 0.2:
        failures.append("VAD flags silence as speech")

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nvoice smoke: ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
