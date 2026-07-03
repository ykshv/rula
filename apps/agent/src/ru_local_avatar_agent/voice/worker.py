"""Per-session LiveKit voice worker.

One worker joins the session's LiveKit room as the avatar participant and runs
the full hot path in-process:

    mic track -> VAD -> turn policy -> (partial) GigaAM STT
        -> speculative/committed Qwen LLM stream -> clause chunker
        -> Qwen3-TTS -> LiveKit audio track
                     -> Audio2Face-3D -> blendshape envelopes (data channel)

Every outbound artifact carries session_id/turn_id/generation_id/branch_state/
seq (and pts_ms where applicable). Interrupt bumps generation_id; anything
stale is dropped silently.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from ru_local_avatar_agent.domain.events import BranchState, EventKind, StreamEnvelope, TurnContext
from ru_local_avatar_agent.domain.session import InvalidTransition, SessionStateMachine
from ru_local_avatar_agent.runtime.clause_chunker import ClauseChunker
from ru_local_avatar_agent.runtime.vllm_client import DialogueModelUnavailable
from ru_local_avatar_agent.voice import metrics
from ru_local_avatar_agent.voice.audio import int16_to_float32, resample
from ru_local_avatar_agent.voice.audit import ConversationAuditEvent
from ru_local_avatar_agent.voice.barge import verify_barge_in_transcript
from ru_local_avatar_agent.voice.brain import ConversationBrain
from ru_local_avatar_agent.voice.face import A2F_SAMPLE_RATE, A2FTurnStream
from ru_local_avatar_agent.voice.face_timeline import face_emit_horizon_ms
from ru_local_avatar_agent.voice.text_match import speculative_transcript_matches
from ru_local_avatar_agent.voice.trace import TurnTrace
from ru_local_avatar_agent.voice.tts_units import split_tts_units
from ru_local_avatar_agent.voice.turn import TurnPolicy, TurnSignal
from ru_local_avatar_agent.voice.vad import VAD_FRAME_SAMPLES

if TYPE_CHECKING:
    from ru_local_avatar_agent.voice.runtime import VoiceRuntime

logger = logging.getLogger(__name__)

DATA_TOPIC = "avatar-protocol"
MIC_SAMPLE_RATE = 16_000
PREROLL_FRAMES = 20  # ~640 ms of audio kept before detected speech start
PARTIAL_INTERVAL_MS = 160
# Short source queue: after a duck the already-queued frames keep playing,
# so this bounds how long the avatar keeps talking over the user (~120 ms
# plus network jitter). The pacer's own buffer provides smoothing.
SOURCE_QUEUE_MS = 120
IDLE_TIMEOUT_S = 300
PARTICIPANT_RECONNECT_GRACE_S = 12
BARGE_CONFIRM_MIN_MS = 560
BARGE_REJECT_COOLDOWN_MS = 700


class VoiceSessionWorker:
    def __init__(
        self,
        runtime: VoiceRuntime,
        *,
        session: SessionStateMachine,
        room_name: str,
        barge_in_enabled: bool = True,
    ) -> None:
        self.runtime = runtime
        self.session = session
        self.room_name = room_name
        self.barge_in_enabled = barge_in_enabled
        self._room = None
        self._source = None
        self._closed = asyncio.Event()
        self._policy = TurnPolicy(tuning=runtime.turn_tuning)
        self._vad_state = runtime.create_vad()

        self._mic_task: asyncio.Task | None = None
        self._partial_task: asyncio.Task | None = None
        self._generation_task: asyncio.Task | None = None
        self._participant_grace_task: asyncio.Task | None = None

        self._utterance: list[np.ndarray] = []
        self._preroll: list[np.ndarray] = []
        self._utterance_lock = asyncio.Lock()

        self._latest_partial = ""
        self._partial_covered_samples = 0
        self._voiced_samples = 0
        self._speculative_text = ""
        self._speculation_wanted = False
        self._gate = asyncio.Event()

        self._history: list[dict[str, str]] = []
        self._brain = ConversationBrain()
        self._pending_prefix = ""
        self._last_final_text = ""
        self._active_assistant_text = ""
        self._last_commit_wall = 0.0
        self._first_audio_wall: float | None = None
        self._last_voiced_wall: float | None = None
        self._speech_end_wall: float | None = None
        self._barge_onset_wall: float | None = None
        self._barge_rejected_until_wall = 0.0
        self._active_pacer = None
        self._ducked = False
        self._barge_confirm_task: asyncio.Task | None = None
        self._any_voiced_wall = 0.0
        self._last_speech_finished_wall = 0.0
        self._speaking = False
        self._last_activity = time.monotonic()
        self._logged_first_mic_frame = False
        self._logged_first_voiced_frame = False
        self._trace_marks: dict[int, dict[str, float]] = {}

    # ------------------------------------------------------------- lifecycle

    async def run(self) -> None:
        from livekit import rtc

        token = self.runtime.mint_token(
            identity="avatar-agent",
            name="Avatar Agent",
            room_name=self.room_name,
        )
        room = rtc.Room()
        self._room = room
        mic_frames: asyncio.Queue = asyncio.Queue(maxsize=256)

        @room.on("track_subscribed")
        def _on_track(track, publication, participant):
            self._cancel_participant_grace()
            logger.info(
                "track subscribed room=%s participant=%s kind=%s",
                self.room_name,
                getattr(participant, "identity", "unknown"),
                track.kind,
            )
            if track.kind == rtc.TrackKind.KIND_AUDIO and self._mic_task is None:
                stream = rtc.AudioStream.from_track(
                    track=track, sample_rate=MIC_SAMPLE_RATE, num_channels=1
                )
                self._mic_task = asyncio.create_task(self._pump_mic(stream, mic_frames))

        @room.on("participant_disconnected")
        def _on_leave(participant):
            if participant.identity != "avatar-agent":
                logger.info(
                    "participant disconnected room=%s participant=%s; waiting %.1fs for reconnect",
                    self.room_name,
                    participant.identity,
                    PARTICIPANT_RECONNECT_GRACE_S,
                )
                self._schedule_participant_grace_close()

        @room.on("disconnected")
        def _on_disconnected(*args):
            logger.info("voice worker disconnected from room %s", self.room_name)
            self._closed.set()

        await room.connect(
            self.runtime.livekit_ws_url,
            token,
            options=rtc.RoomOptions(
                auto_subscribe=True,
                connect_timeout=30,
                single_peer_connection=False,
            ),
        )
        logger.info("voice worker joined room %s", self.room_name)

        self._source = rtc.AudioSource(
            self.runtime.tts_sample_rate, 1, queue_size_ms=SOURCE_QUEUE_MS
        )
        track = rtc.LocalAudioTrack.create_audio_track("avatar-voice", self._source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await room.local_participant.publish_track(track, options)

        metrics.ACTIVE_SESSIONS.inc()
        try:
            await self._send_state("idle")
            await self._main_loop(mic_frames)
        finally:
            metrics.ACTIVE_SESSIONS.dec()
            await self._cancel_generation("worker_shutdown")
            for task in (self._mic_task, self._partial_task):
                if task is not None:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            if self._participant_grace_task is not None:
                self._participant_grace_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._participant_grace_task
            with contextlib.suppress(Exception):
                await room.disconnect()
            logger.info("voice worker for %s stopped", self.room_name)

    def close(self) -> None:
        self._closed.set()

    def _cancel_participant_grace(self) -> None:
        if self._participant_grace_task is not None and not self._participant_grace_task.done():
            self._participant_grace_task.cancel()
        self._participant_grace_task = None

    def _schedule_participant_grace_close(self) -> None:
        self._cancel_participant_grace()
        self._participant_grace_task = asyncio.create_task(self._participant_grace_close())

    async def _participant_grace_close(self) -> None:
        try:
            await asyncio.sleep(PARTICIPANT_RECONNECT_GRACE_S)
        except asyncio.CancelledError:
            raise
        logger.info("participant reconnect grace expired room=%s; closing worker", self.room_name)
        self._closed.set()

    async def _pump_mic(self, stream, out: asyncio.Queue) -> None:
        async for event in stream:
            frame = event.frame
            pcm = np.frombuffer(frame.data, dtype=np.int16)
            if not self._logged_first_mic_frame:
                self._logged_first_mic_frame = True
                logger.info(
                    "first mic frame room=%s samples=%d sample_rate=%s",
                    self.room_name,
                    pcm.shape[0],
                    getattr(frame, "sample_rate", "unknown"),
                )
            with contextlib.suppress(asyncio.QueueFull):
                out.put_nowait(pcm)

    # ------------------------------------------------------------- main loop

    async def _main_loop(self, mic_frames: asyncio.Queue) -> None:
        pending = np.empty(0, dtype=np.int16)
        while not self._closed.is_set():
            try:
                pcm = await asyncio.wait_for(mic_frames.get(), timeout=1.0)
            except TimeoutError:
                if time.monotonic() - self._last_activity > IDLE_TIMEOUT_S:
                    logger.info("room %s idle for %ss, closing", self.room_name, IDLE_TIMEOUT_S)
                    break
                continue
            pending = np.concatenate([pending, pcm])
            while pending.shape[0] >= VAD_FRAME_SAMPLES:
                frame = pending[:VAD_FRAME_SAMPLES]
                pending = pending[VAD_FRAME_SAMPLES:]
                await self._process_vad_frame(frame)

    async def _process_vad_frame(self, frame_i16: np.ndarray) -> None:
        frame = int16_to_float32(frame_i16)
        probability = await asyncio.to_thread(self._vad_state.probability, frame)
        now = time.monotonic()

        agent_speaking = self._speaking or self.session.branch_state in {
            BranchState.COMMITTED,
            BranchState.SPEAKING,
        }
        if agent_speaking and not self.barge_in_enabled:
            self._utterance = []
            self._barge_onset_wall = None
            self._policy.reset_barge()
            return
        voiced = probability >= self._policy.tuning.voiced_on_probability
        early_playback_echo = (
            agent_speaking
            and self._first_audio_wall is not None
            and (now - self._first_audio_wall) * 1000
            < self._policy.tuning.barge_in_min_speaking_ms
        )
        barge_reject_cooldown = now < self._barge_rejected_until_wall
        barge_candidate = (
            probability >= self._policy.tuning.barge_in_probability
            and not early_playback_echo
            and not barge_reject_cooldown
        )
        if voiced:
            if not self._logged_first_voiced_frame:
                self._logged_first_voiced_frame = True
                logger.info(
                    "first voiced frame room=%s probability=%.3f",
                    self.room_name,
                    probability,
                )
            self._last_activity = now
            self._any_voiced_wall = now
            if agent_speaking and barge_candidate and self._barge_onset_wall is None:
                self._barge_onset_wall = now
            if not agent_speaking:
                self._last_voiced_wall = now
        if not agent_speaking or (not barge_candidate and not self._ducked):
            self._barge_onset_wall = None

        if agent_speaking:
            if self._ducked:
                # While ducked we keep everything (speech + natural pauses):
                # this buffer becomes the STT input that confirms the barge
                # and, when confirmed, the user's actual utterance.
                self._utterance.append(frame)
            elif barge_candidate:
                self._utterance.append(frame)
            else:
                self._utterance = []
        elif self._policy.utterance_active or voiced:
            self._utterance.append(frame)
            if voiced and not agent_speaking:
                self._voiced_samples = sum(f.shape[0] for f in self._utterance)
        else:
            self._preroll.append(frame)
            if len(self._preroll) > PREROLL_FRAMES:
                self._preroll.pop(0)

        policy_probability = 0.0 if (early_playback_echo or barge_reject_cooldown) else probability
        for signal in self._policy.feed(policy_probability, agent_speaking=agent_speaking):
            await self._handle_signal(signal)

    # ------------------------------------------------------------ signals

    async def _handle_signal(self, signal: TurnSignal) -> None:
        if signal == TurnSignal.USER_SPEECH_START:
            self._utterance = list(self._preroll) + self._utterance
            self._preroll = []
            self._latest_partial = ""
            self._partial_covered_samples = 0
            self._voiced_samples = sum(f.shape[0] for f in self._utterance)
            if self._partial_task is None or self._partial_task.done():
                self._partial_task = asyncio.create_task(self._partial_loop())
            await self._send_state("listening")
        elif signal == TurnSignal.SPECULATE:
            await self._start_speculative()
        elif signal == TurnSignal.DISCARD:
            await self._discard_speculative("user_resumed")
        elif signal == TurnSignal.COMMIT:
            await self._commit()
        elif signal == TurnSignal.BARGE_DUCK:
            self._duck_playback()
        elif signal == TurnSignal.BARGE_ABORT:
            # While the transcript check is in flight it owns the decision;
            # a quiet gap alone must not resume over a confirmed interrupt.
            if self._barge_confirm_task is None or self._barge_confirm_task.done():
                self._resume_playback("vad_quiet")
        elif signal == TurnSignal.BARGE_IN:
            if self._ducked or (
                self._barge_confirm_task is not None and not self._barge_confirm_task.done()
            ):
                return  # duck flow already verifying
            await self._confirm_or_reject_barge_in()

    async def _partial_loop(self) -> None:
        while self._policy.utterance_active and not self._closed.is_set():
            snapshot = self._utterance_snapshot()
            # Transcribe only when there is uncovered VOICED audio (plus a
            # short tail). During post-speech silence this yields exactly one
            # final partial and then frees the GPU for the speculative
            # LLM/TTS — running partials over growing silence starved the
            # first-audio path.
            needs_partial = (
                self._partial_covered_samples
                < self._voiced_samples + int(0.15 * MIC_SAMPLE_RATE)
            )
            if not needs_partial:
                await asyncio.sleep(PARTIAL_INTERVAL_MS / 1000)
                continue
            if snapshot.shape[0] >= VAD_FRAME_SAMPLES * 4:
                started = time.monotonic()
                try:
                    text = await self.runtime.run_stt(snapshot)
                except Exception:
                    logger.exception("partial STT failed")
                    metrics.PIPELINE_ERRORS.labels(stage="stt_partial").inc()
                    text = ""
                elapsed = time.monotonic() - started
                metrics.STT_PARTIAL_MS.observe(elapsed * 1000)
                if text and self._policy.utterance_active:
                    self._partial_covered_samples = int(snapshot.shape[0])
                    # Always refresh the policy: the post-silence partial is
                    # usually IDENTICAL to the last in-speech one, and it is
                    # exactly the one that must arm the punctuated fast path.
                    self._policy.note_partial(text)
                    if text != self._latest_partial:
                        self._latest_partial = text
                        await self._send_envelope(
                            self.session.current_context(),
                            kind=EventKind.PARTIAL_TRANSCRIPT,
                            payload={"text": text},
                        )
                    # A SPECULATE signal may have fired before any partial
                    # existed; start the speculation as soon as text arrives.
                    if self._speculation_wanted:
                        self._speculation_wanted = False
                        await self._start_speculative()
                    elif (
                        self.session.branch_state == BranchState.SPECULATIVE
                        and self._speculative_text
                        and not speculative_transcript_matches(
                            self._speculative_text, text
                        )
                    ):
                        # The post-silence partial diverged from what we are
                        # speculating on — a guaranteed miss at commit time.
                        # Restart now so the fresh answer is ready by commit
                        # instead of paying the full serial path afterwards.
                        logger.info(
                            "re-speculating: partial diverged from %r",
                            self._speculative_text[:48],
                        )
                        await self._discard_speculative("partial_diverged")
                        await self._start_speculative()
                # Keep the effective cadence close to the target interval
                # instead of interval + STT wall time.
                await asyncio.sleep(max(0.02, PARTIAL_INTERVAL_MS / 1000 - elapsed))
                continue
            await asyncio.sleep(PARTIAL_INTERVAL_MS / 1000)

    def _utterance_snapshot(self) -> np.ndarray:
        if not self._utterance:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self._utterance)

    async def _start_speculative(self) -> None:
        if self.session.branch_state != BranchState.LISTENING:
            return
        text = self._latest_partial.strip()
        if not text:
            # No transcript yet — retry from the partial loop when it lands.
            self._speculation_wanted = True
            return
        if self._pending_prefix:
            text = f"{self._pending_prefix} {text}".strip()
        ctx = self.session.start_speculative("silence_speculation")
        self._speculative_text = text
        self._gate = asyncio.Event()
        metrics.SPECULATIVE_STARTS.inc()
        self._trace_marks.setdefault(ctx.generation_id, {})["speculation_start"] = time.monotonic()
        logger.info("speculation started gen=%d text=%r", ctx.generation_id, text[:60])
        await self._send_state("thinking", ctx=ctx)
        self._generation_task = asyncio.create_task(self._generate(ctx, text))

    async def _discard_speculative(self, reason: str) -> None:
        if self.session.branch_state != BranchState.SPECULATIVE:
            return
        metrics.SPECULATIVE_DISCARDS.inc()
        self.session.discard_speculative(self.session.generation_id, reason)
        await self._cancel_generation(reason)
        await self._send_state("listening")

    async def _commit(self) -> None:
        speech_end = self._last_voiced_wall or time.monotonic()
        audio = self._utterance_snapshot()
        self._utterance = []
        if audio.shape[0] < VAD_FRAME_SAMPLES * 4:
            await self._discard_speculative("too_short")
            return

        # Reuse the latest partial only when it already covers the full
        # utterance (the tail past coverage must be silence, not speech).
        coverage_ok = self._partial_covered_samples >= audio.shape[0] - int(0.30 * MIC_SAMPLE_RATE)
        final_text = self._latest_partial.strip() if coverage_ok else ""
        if final_text:
            logger.info("commit reusing latest partial transcript chars=%d", len(final_text))
        else:
            started = time.monotonic()
            try:
                final_text = await self.runtime.run_stt(audio)
            except Exception:
                logger.exception("final STT failed")
                metrics.PIPELINE_ERRORS.labels(stage="stt_final").inc()
                final_text = ""
            metrics.STT_FINAL_MS.observe((time.monotonic() - started) * 1000)

        if not final_text.strip():
            await self._discard_speculative("empty_transcript")
            self._policy.reset_turn()
            await self._send_state("idle")
            return

        # Echo guard: right after the avatar finishes talking, its own
        # speaker tail can leak through the mic and produce a "user" turn
        # that is really the avatar's sentence. Attributing it to the user
        # is exactly the "who said what" confusion — drop it instead.
        if time.monotonic() - self._last_speech_finished_wall < 3.0:
            reference = self._brain.state.last_assistant_final or self._active_assistant_text
            if reference:
                echo = verify_barge_in_transcript(final_text, reference)
                if not echo.confirmed and echo.reason.startswith("assistant_echo"):
                    logger.info(
                        "dropping post-playback echo as user turn: %r (%s)",
                        final_text[:60],
                        echo.reason,
                    )
                    metrics.ECHO_TURN_DROPS.inc()
                    await self._discard_speculative("playback_echo")
                    self._policy.reset_turn()
                    await self._send_state("idle")
                    return

        await self._commit_transcript(final_text, speech_end=speech_end)

    async def _commit_transcript(self, final_text: str, *, speech_end: float) -> None:
        """Commit a finished user utterance and start (or release) the answer.

        Shared by the VAD-driven commit path and the confirmed barge-in path,
        which already holds a transcript of the interrupting speech.
        """
        # A barge-in right after a commit usually means the previous EOT was
        # premature (a mid-sentence pause GigaAM punctuated as complete);
        # carry the cut-off transcript so the question keeps its context.
        if self._pending_prefix:
            final_text = f"{self._pending_prefix} {final_text}".strip()
            self._pending_prefix = ""

        speculative_alive = (
            self.session.branch_state == BranchState.SPECULATIVE
            and self._generation_task is not None
            and not self._generation_task.done()
        )
        speculative_hit = speculative_alive and speculative_transcript_matches(
            self._speculative_text,
            final_text,
        )

        try:
            ctx = self.session.commit_eot("vad_eot")
        except InvalidTransition:
            logger.warning("commit in unexpected state %s", self.session.branch_state)
            return
        metrics.EOT_COMMIT_MS.observe((time.monotonic() - speech_end) * 1000)
        self._last_commit_wall = time.monotonic()
        self._last_final_text = final_text
        self._trace_marks.setdefault(ctx.generation_id, {})["speech_end"] = speech_end
        self._trace_marks.setdefault(ctx.generation_id, {})["eot_commit"] = self._last_commit_wall
        self._brain.record_user_turn(
            final_text,
            turn_id=ctx.turn_id,
            generation_id=ctx.generation_id,
        )
        self._audit(ctx, "user_final", final_text)
        self._audit(ctx, "state_update", payload={"state": self._brain.state.snapshot()})

        await self._send_envelope(
            ctx,
            kind=EventKind.FINAL_TRANSCRIPT,
            payload={"text": final_text},
        )

        self._speculation_wanted = False
        logger.info(
            "commit gen=%d hit=%s text=%r",
            ctx.generation_id,
            speculative_hit,
            final_text[:60],
        )
        if speculative_hit:
            metrics.SPECULATIVE_HITS.inc()
            self._speech_end_wall = speech_end
            self._gate.set()
        else:
            if speculative_alive:
                metrics.SPECULATIVE_DISCARDS.inc()
            await self._cancel_generation("speculation_mismatch")
            self._speech_end_wall = speech_end
            self._gate = asyncio.Event()
            self._gate.set()
            self._generation_task = asyncio.create_task(self._generate(ctx, final_text))
        await self._send_state("thinking", ctx=self.session.current_context())

    def _duck_playback(self) -> None:
        """Pause the avatar's voice the moment the user plausibly starts talking.

        Cheap and reversible: buffered PCM is kept, only frame pushing stops,
        so a rejected barge resumes mid-word without losing audio. The short
        LiveKit source queue bounds the audible tail after the pause.
        """
        pacer = self._active_pacer
        if pacer is None or self._ducked or not self._speaking:
            return
        self._ducked = True
        pacer.pause()
        metrics.BARGE_DUCKS.inc()
        onset = self._barge_onset_wall
        if onset is not None:
            metrics.BARGE_IN_MS.observe((time.monotonic() - onset) * 1000)
        logger.info("playback ducked gen=%d", self.session.generation_id)
        # Kick off transcript verification right away: a confirmed interrupt
        # cancels the answer AND commits the user's interrupting utterance,
        # so short interjections ("стоп", a new question) are never lost.
        if self._barge_confirm_task is None or self._barge_confirm_task.done():
            self._barge_confirm_task = asyncio.create_task(self._barge_confirm_flow())

    def _resume_playback(self, reason: str) -> None:
        pacer = self._active_pacer
        if pacer is None or not self._ducked:
            return
        self._ducked = False
        pacer.resume()
        metrics.BARGE_DUCK_ABORTS.inc()
        self._policy.reset_barge()
        self._barge_onset_wall = None
        logger.info("playback resumed gen=%d reason=%s", self.session.generation_id, reason)

    async def _barge_confirm_flow(self) -> None:
        """Verify a duck with STT; confirmed -> full barge + commit the speech.

        Runs concurrently with the paused answer. Rejected (echo/noise) ->
        playback resumes exactly where it stopped.
        """
        min_samples = int(0.36 * MIC_SAMPLE_RATE)
        deadline = time.monotonic() + 1.6
        while time.monotonic() < deadline:
            if not self._ducked:
                return  # resumed by the quiet-abort path
            audio = self._utterance_snapshot()
            quiet_for = time.monotonic() - self._any_voiced_wall
            if audio.shape[0] >= min_samples and (
                quiet_for > 0.30 or audio.shape[0] >= int(1.2 * MIC_SAMPLE_RATE)
            ):
                break
            await asyncio.sleep(0.05)

        audio = self._utterance_snapshot()
        if audio.shape[0] < min_samples // 2:
            self._resume_playback("confirm_too_short")
            return
        try:
            transcript = await self.runtime.run_stt(audio)
        except Exception:
            logger.exception("barge-in confirmation STT failed")
            metrics.PIPELINE_ERRORS.labels(stage="barge_stt").inc()
            transcript = ""
        if not self._ducked:
            return

        decision = verify_barge_in_transcript(transcript, self._active_assistant_text)
        if not decision.confirmed:
            logger.info(
                "duck rejected reason=%s transcript=%r",
                decision.reason,
                decision.transcript[:80],
            )
            metrics.BARGE_IN_REJECTS.labels(reason=decision.reason).inc()
            self._barge_rejected_until_wall = time.monotonic() + BARGE_REJECT_COOLDOWN_MS / 1000
            self._utterance = []
            self._resume_playback(f"rejected:{decision.reason}")
            return

        logger.info(
            "duck confirmed reason=%s transcript=%r",
            decision.reason,
            decision.transcript[:80],
        )
        still_talking = time.monotonic() - self._any_voiced_wall < 0.32
        await self._barge_in()
        if still_talking:
            # The user keeps talking: restore the full buffer so the normal
            # partial/commit machinery continues from what we already heard.
            self._preroll = []
            self._utterance = [audio]
            self._latest_partial = transcript.strip()
            self._partial_covered_samples = int(audio.shape[0])
            self._voiced_samples = int(audio.shape[0])
            self._policy.note_partial(self._latest_partial)
        else:
            # The interrupting utterance is already complete: answer it now.
            self._utterance = []
            await self._commit_transcript(
                transcript.strip(),
                speech_end=self._any_voiced_wall or time.monotonic(),
            )

    async def _barge_in(self) -> None:
        onset = self._barge_onset_wall or time.monotonic()
        # _cancel_generation resets _ducked via the generation's finally
        # block, so capture it now for the metric decision below.
        was_ducked = self._ducked
        # Carry the previous transcript only when the interrupted answer had
        # barely (or not yet) started sounding — that pattern means our EOT
        # was premature and the user is finishing the SAME sentence. A barge
        # midway through an audible answer is a new intent, not a continuation.
        now = time.monotonic()
        answer_barely_started = (
            self._first_audio_wall is None or now - self._first_audio_wall < 0.4
        )
        if (
            now - self._last_commit_wall < 2.5
            and self._last_final_text
            and answer_barely_started
        ):
            self._pending_prefix = self._last_final_text
        self._speculation_wanted = False
        logger.info(
            "barge-in gen=%d prefix_carried=%s",
            self.session.generation_id,
            bool(self._pending_prefix),
        )
        interrupted_text = self._active_assistant_text.strip()
        if interrupted_text:
            ctx = self.session.current_context()
            self._brain.record_assistant_interrupted(
                interrupted_text,
                turn_id=ctx.turn_id,
                generation_id=ctx.generation_id,
            )
            self._audit(ctx, "assistant_interrupted", interrupted_text)
            self._audit(ctx, "state_update", payload={"state": self._brain.state.snapshot()})
        metrics.INTERRUPTS.inc()
        self.session.interrupt("barge_in")
        await self._cancel_generation("barge_in")
        if self._source is not None:
            self._source.clear_queue()
        self._speaking = False
        self._barge_onset_wall = None
        # Audible stop already happened at duck time; this histogram then
        # tracks the full transcript-confirmed cancellation instead.
        if not was_ducked:
            metrics.BARGE_IN_MS.observe((time.monotonic() - onset) * 1000)
        self._ducked = False
        metrics.BARGE_CONFIRM_MS.observe((time.monotonic() - onset) * 1000)
        self._policy.reset_turn()
        # Keep the speech that triggered the barge-in as preroll so the new
        # utterance does not lose its onset.
        self._preroll = (self._preroll + self._utterance)[-PREROLL_FRAMES:]
        self._utterance = []
        await self._send_state("interrupted")
        await self._send_state("listening")

    async def _confirm_or_reject_barge_in(self) -> None:
        audio = self._utterance_snapshot()
        min_samples = MIC_SAMPLE_RATE * BARGE_CONFIRM_MIN_MS // 1000
        if audio.shape[0] < min_samples:
            self._policy.reset_barge()
            return

        try:
            transcript = await self.runtime.run_stt(audio)
        except Exception:
            logger.exception("barge-in confirmation STT failed")
            metrics.PIPELINE_ERRORS.labels(stage="barge_stt").inc()
            transcript = ""

        decision = verify_barge_in_transcript(transcript, self._active_assistant_text)
        if decision.confirmed:
            logger.info(
                "barge-in confirmed reason=%s transcript=%r",
                decision.reason,
                decision.transcript[:80],
            )
            await self._barge_in()
            return

        logger.info(
            "barge-in rejected reason=%s transcript=%r assistant=%r",
            decision.reason,
            decision.transcript[:80],
            self._active_assistant_text[:80],
        )
        metrics.BARGE_IN_REJECTS.labels(reason=decision.reason).inc()
        self._barge_onset_wall = None
        self._barge_rejected_until_wall = time.monotonic() + BARGE_REJECT_COOLDOWN_MS / 1000
        self._utterance = []
        self._policy.reset_barge()
        self._resume_playback(f"rejected:{decision.reason}")

    async def _cancel_generation(self, reason: str) -> None:
        task = self._generation_task
        self._generation_task = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._speculative_text = ""
        self._active_assistant_text = ""

    # ------------------------------------------------------------ generation

    async def _generate(self, ctx: TurnContext, user_text: str) -> None:
        """Stream LLM -> clause TTS -> paced audio + face side channel.

        The audio hot path is owned by AudioPacer; face inference and
        data-channel sends run in a parallel task so reliable-channel latency
        can never starve playback (that starvation was audible as mid-word
        stutter).
        """
        from ru_local_avatar_agent.voice.audio_pacer import AudioPacer

        seq = 0
        assistant_text: list[str] = []
        model = self.runtime.dialogue_model
        chunker = ClauseChunker(token_target=self.runtime.first_chunk_token_target)
        plan = self._brain.plan_response(self._history, user_text)
        initial_tts_text = plan.direct_text if plan.mode == "direct" else ""
        playback_settings = self.runtime.playback_settings_for_text(
            initial_tts_text,
            planned_latency_tier=plan.latency_tier,
            planned_playback_policy=plan.playback_policy,
        )
        trace_marks = self._trace_marks.setdefault(ctx.generation_id, {})
        trace = TurnTrace(
            session_id=ctx.session_id,
            turn_id=ctx.turn_id,
            generation_id=ctx.generation_id,
            speech_end_wall=trace_marks.get("speech_end", self._speech_end_wall),
            latency_tier=playback_settings.latency_tier,
            playback_policy=playback_settings.playback_policy,
        )
        for mark_name, mark_wall in trace_marks.items():
            trace.mark(mark_name, mark_wall)
        face_stream = A2FTurnStream(self.runtime.face_engine)
        face_queue: asyncio.Queue = asyncio.Queue()
        submitted_ms = 0.0
        pending_speech_segments: list[tuple[int, dict]] = []
        self._first_audio_wall = None
        self._active_assistant_text = ""
        llm_started = time.monotonic()
        first_token_at: float | None = None
        tts_sr = self.runtime.tts_sample_rate

        def stale() -> bool:
            return self.session.generation_id != ctx.generation_id

        def sync_trace_marks() -> None:
            marks = self._trace_marks.get(ctx.generation_id, {})
            if "speech_end" in marks:
                trace.speech_end_wall = marks["speech_end"]
            for mark_name, mark_wall in marks.items():
                trace.mark(mark_name, mark_wall)

        async def on_first_frame() -> None:
            self._speaking = True
            self._first_audio_wall = time.monotonic()
            with contextlib.suppress(InvalidTransition):
                self.session.start_speaking()
            if self._speech_end_wall is not None:
                first_audio_ms = (time.monotonic() - self._speech_end_wall) * 1000
                metrics.FIRST_AUDIO_MS.observe(first_audio_ms)
                metrics.FIRST_AUDIO_BY_TIER_MS.labels(
                    latency_tier=trace.latency_tier,
                ).observe(first_audio_ms)
                logger.info("first audio gen=%d after %.0f ms", ctx.generation_id, first_audio_ms)
            trace.mark("first_audio", self._first_audio_wall)
            await self._send_state("speaking", ctx=ctx)

        logger.info(
            (
                "audio playback mode gen=%d tier=%s policy=%s cache_hit=%s "
                "prebuffer_ms=%d rebuffer_ms=%d unit_start_min_ms=%d speculative_tts=%s"
            ),
            ctx.generation_id,
            playback_settings.latency_tier,
            playback_settings.playback_policy,
            playback_settings.cache_hit,
            playback_settings.prebuffer_ms,
            playback_settings.rebuffer_ms,
            playback_settings.unit_start_min_ms,
            self.runtime.tts_speculative_enabled,
        )
        pacer = AudioPacer(
            self._source,
            tts_sr,
            prebuffer_ms=playback_settings.prebuffer_ms,
            rebuffer_ms=playback_settings.rebuffer_ms,
            unit_start_min_ms=playback_settings.unit_start_min_ms,
            start_after_end=playback_settings.start_after_response,
            on_first_frame=on_first_frame,
        )
        self._active_pacer = pacer
        self._ducked = False
        face_task = asyncio.create_task(
            self._face_sender(ctx, face_stream, face_queue, pacer)
        )

        async def send_or_buffer_speech_segment(*, pts_ms: int, payload: dict) -> None:
            nonlocal seq
            if playback_settings.start_after_response:
                pending_speech_segments.append((pts_ms, payload))
                return
            seq += 1
            await self._send_envelope(
                ctx,
                kind=EventKind.SPEECH_SEGMENT,
                seq=seq,
                pts_ms=pts_ms,
                payload=payload,
            )

        async def flush_pending_speech_segments() -> None:
            nonlocal seq
            if not pending_speech_segments:
                return
            for pts_ms, payload in pending_speech_segments:
                if stale():
                    metrics.STALE_DROPS.inc()
                    trace.stale_drops += 1
                    return
                seq += 1
                await self._send_envelope(
                    ctx,
                    kind=EventKind.SPEECH_SEGMENT,
                    seq=seq,
                    pts_ms=pts_ms,
                    payload=payload,
                )
            pending_speech_segments.clear()

        async def push_clause(clause_text: str, *, reason: str) -> None:
            """Stream one logical clause through one or more TTS-safe units."""
            nonlocal seq, submitted_ms
            speculative_tts_blocked = (
                ctx.branch_state == BranchState.SPECULATIVE
                and not self.runtime.tts_speculative_enabled
            )
            if speculative_tts_blocked:
                await self._gate.wait()
                sync_trace_marks()
                if stale():
                    metrics.STALE_DROPS.inc()
                    trace.stale_drops += 1
                    return
            logger.info(
                "tts clause gen=%d reason=%s chars=%d text=%r",
                ctx.generation_id,
                reason,
                len(clause_text),
                clause_text[:160],
            )
            units = split_tts_units(clause_text)
            for unit_index, unit_text in enumerate(units):
                logger.info(
                    "tts unit gen=%d unit=%d/%d chars=%d text=%r",
                    ctx.generation_id,
                    unit_index + 1,
                    len(units),
                    len(unit_text),
                    unit_text[:160],
                )
                synth_started = time.monotonic()
                unit_cache_hit = self.runtime.is_tts_cached(unit_text)
                cache_state = "hit" if unit_cache_hit else "miss"
                trace.observe_tts_unit(cache_hit=unit_cache_hit)
                unit_deadline = synth_started + self.runtime.tts_unit_timeout_ms / 1000
                unit_audio_ms = 0.0
                unit_chunks = 0
                first_chunk_seen = False
                stream = self.runtime.stream_tts(unit_text)
                try:
                    while True:
                        hard_remaining_s = unit_deadline - time.monotonic()
                        if hard_remaining_s <= 0:
                            raise TimeoutError(
                                f"TTS unit exceeded {self.runtime.tts_unit_timeout_ms} ms"
                            )
                        try:
                            tts_chunk = await asyncio.wait_for(
                                stream.__anext__(),
                                timeout=min(
                                    hard_remaining_s,
                                    self.runtime.tts_chunk_timeout_ms / 1000,
                                ),
                            )
                        except StopAsyncIteration:
                            break
                        except TimeoutError as exc:
                            if time.monotonic() >= unit_deadline:
                                raise TimeoutError(
                                    f"TTS unit exceeded {self.runtime.tts_unit_timeout_ms} ms"
                                ) from exc
                            raise TimeoutError(
                                f"TTS chunk gap exceeded {self.runtime.tts_chunk_timeout_ms} ms"
                            ) from exc
                        if stale():
                            metrics.STALE_DROPS.inc()
                            trace.stale_drops += 1
                            return
                        if not first_chunk_seen:
                            first_chunk_seen = True
                            trace.mark("tts_first_chunk")
                            if pacer.first_push_at is None:
                                metrics.TTS_FIRST_CHUNK_MS.observe(
                                    (time.monotonic() - synth_started) * 1000
                                )
                            await send_or_buffer_speech_segment(
                                pts_ms=int(submitted_ms),
                                payload={
                                    "text": unit_text,
                                    "clause_text": clause_text,
                                    "unit_index": unit_index,
                                    "unit_count": len(units),
                                    "latency_tier": trace.latency_tier,
                                    "cache_hit": unit_cache_hit,
                                    "playback_policy": playback_settings.playback_policy,
                                },
                            )
                        if len(tts_chunk.pcm) == 0:
                            continue
                        chunk_ms = len(tts_chunk.pcm) * 1000 / tts_chunk.sample_rate
                        pacer.submit(tts_chunk.pcm)
                        submitted_ms += chunk_ms
                        unit_audio_ms += chunk_ms
                        unit_chunks += 1
                        await face_queue.put(tts_chunk)
                except TimeoutError:
                    metrics.TTS_UNIT_TIMEOUTS.inc()
                    metrics.PIPELINE_ERRORS.labels(stage="tts_unit_timeout").inc()
                    logger.error(
                        "tts unit timed out gen=%d unit=%d/%d timeout_ms=%d text=%r",
                        ctx.generation_id,
                        unit_index + 1,
                        len(units),
                        self.runtime.tts_unit_timeout_ms,
                        unit_text[:160],
                    )
                    raise
                else:
                    unit_wall_ms = (time.monotonic() - synth_started) * 1000
                    metrics.TTS_UNIT_WALL_MS.labels(
                        latency_tier=trace.latency_tier,
                        cache_state=cache_state,
                    ).observe(unit_wall_ms)
                    metrics.TTS_UNIT_AUDIO_MS.labels(
                        latency_tier=trace.latency_tier,
                        cache_state=cache_state,
                    ).observe(unit_audio_ms)
                    logger.info(
                        "tts unit done gen=%d unit=%d/%d chunks=%d audio_ms=%.0f "
                        "wall_ms=%.0f pacer_pushed_ms=%.0f pacer_buffered_ms=%.0f",
                        ctx.generation_id,
                        unit_index + 1,
                        len(units),
                        unit_chunks,
                        unit_audio_ms,
                        unit_wall_ms,
                        pacer.pushed_ms,
                        pacer.buffered_ms,
                    )
                    pacer.notify_unit_complete()
                finally:
                    await stream.aclose()

        try:
            if plan.mode == "direct":
                assistant_text.append(plan.direct_text)
                self._active_assistant_text = plan.direct_text
                await push_clause(plan.direct_text, reason=f"direct:{plan.reason}")
            else:
                async for token in model.stream(plan.messages):
                    if stale():
                        metrics.STALE_DROPS.inc()
                        trace.stale_drops += 1
                        return
                    if token.text:
                        if first_token_at is None:
                            first_token_at = time.monotonic()
                            trace.mark("llm_first_token", first_token_at)
                            metrics.LLM_FIRST_TOKEN_MS.observe(
                                (first_token_at - llm_started) * 1000
                            )
                        assistant_text.append(token.text)
                        self._active_assistant_text = "".join(assistant_text).strip()
                        chunk = chunker.push(token.text)
                        if chunk:
                            await push_clause(chunk.text, reason=chunk.reason)
                            # First clause is latency-critical and short; later
                            # clauses run longer for better prosody.
                            chunker.token_target = self.runtime.later_chunk_token_target
            final_chunk = chunker.finish()
            if final_chunk:
                await push_clause(final_chunk.text, reason=final_chunk.reason)

            await flush_pending_speech_segments()
            pacer.end_of_response()
            await face_queue.put(None)  # face task flushes A2F tail and exits
            if stale():
                return
            played = await pacer.wait_played(expected_audio_ms=submitted_ms)
            trace.mark("audio_done")
            if not played:
                metrics.AUDIO_PLAYOUT_TIMEOUTS.inc()
            with contextlib.suppress(asyncio.CancelledError):
                await face_task
            if stale():
                return
            self._speaking = False
            full_text = "".join(assistant_text).strip()
            if full_text:
                self._history.append({"role": "user", "content": user_text})
                self._history.append({"role": "assistant", "content": full_text})
                self._history = self._history[-24:]
                self._brain.record_assistant_final(
                    full_text,
                    turn_id=ctx.turn_id,
                    generation_id=ctx.generation_id,
                )
                self._audit(ctx, "assistant_final", full_text)
                self._audit(ctx, "state_update", payload={"state": self._brain.state.snapshot()})
            trace.underruns = pacer.underruns
            sync_trace_marks()
            trace_payload = trace.summary()
            self._audit(ctx, "turn_trace", payload=trace_payload)
            await self._send_envelope(
                ctx,
                kind=EventKind.TURN_TRACE,
                payload=trace_payload,
            )
            with contextlib.suppress(InvalidTransition):
                self.session.finish_turn("response_complete")
            metrics.TURNS_COMPLETED.inc()
            await self._send_state("idle")
        except asyncio.CancelledError:
            self._speaking = False
            raise
        except DialogueModelUnavailable as exc:
            self._speaking = False
            metrics.PIPELINE_ERRORS.labels(stage="llm").inc()
            logger.error("dialogue model unavailable: %s", exc)
            with contextlib.suppress(InvalidTransition):
                self.session.interrupt("llm_unavailable")
            await self._send_envelope(
                self.session.current_context(),
                kind=EventKind.ERROR,
                payload={"message": "llm_unavailable"},
            )
            await self._send_state("idle")
        except Exception:
            self._speaking = False
            metrics.PIPELINE_ERRORS.labels(stage="generation").inc()
            logger.exception("generation pipeline failed")
            with contextlib.suppress(InvalidTransition):
                self.session.interrupt("pipeline_error")
            await self._send_state("idle")
        finally:
            if pacer.underruns:
                metrics.AUDIO_UNDERRUNS.inc(pacer.underruns)
                logger.warning("turn had %d audio underruns", pacer.underruns)
            if self._active_pacer is pacer:
                self._active_pacer = None
                self._ducked = False
            if not face_task.done():
                face_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await face_task
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pacer.stop(flush_queue=False)
            if not self._speaking:
                self._active_assistant_text = ""

    async def _face_sender(
        self,
        ctx: TurnContext,
        face_stream: A2FTurnStream,
        queue: asyncio.Queue,
        pacer,
    ) -> None:
        """Runs A2F per TTS chunk and paces blendshape envelopes off the audio path."""
        pending: list = []
        flushing = False
        seq = 0
        while True:
            if not flushing:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.08)
                except TimeoutError:
                    item = ...
                if item is None:
                    tail = await self.runtime.run_face(face_stream.flush)
                    pending.extend(tail)
                    flushing = True
                elif item is not ...:
                    face_audio = await asyncio.to_thread(
                        resample, item.pcm, item.sample_rate, A2F_SAMPLE_RATE
                    )
                    frames = await self.runtime.run_face(face_stream.push_audio, face_audio)
                    pending.extend(frames)

            # Once TTS has ended and A2F has flushed its tail, the audio
            # timeline will not advance anymore. Drain pending face frames
            # immediately; otherwise frames beyond the fixed lead horizon keep
            # this task alive until the room disconnects.
            horizon = face_emit_horizon_ms(pushed_ms=pacer.pushed_ms, flushing=flushing)
            batch: list = []
            while pending and pending[0].pts_ms <= horizon:
                batch.append(pending.pop(0))
                if len(batch) >= 6:
                    seq += 1
                    await self._send_face_batch(ctx, batch, seq)
                    batch = []
            if batch:
                seq += 1
                await self._send_face_batch(ctx, batch, seq)
            if flushing:
                if not pending:
                    return
                await asyncio.sleep(0.06)

    async def _send_face_batch(self, ctx: TurnContext, batch: list, seq: int) -> None:
        await self._send_envelope(
            ctx,
            kind=EventKind.AVATAR_BLENDSHAPE_FRAME,
            seq=seq,
            pts_ms=batch[0].pts_ms,
            payload={
                "frames": [{"pts_ms": f.pts_ms, "values": f.values} for f in batch],
            },
        )

    # ------------------------------------------------------------ envelopes

    async def _send_state(self, state: str, ctx: TurnContext | None = None) -> None:
        await self._send_envelope(
            ctx or self.session.current_context(),
            kind=EventKind.AVATAR_STATE,
            payload={"state": state},
        )

    def _audit(
        self,
        ctx: TurnContext,
        event_type: str,
        text: str = "",
        payload: dict | None = None,
    ) -> None:
        try:
            self.runtime.conversation_audit.append(
                ConversationAuditEvent(
                    session_id=ctx.session_id,
                    turn_id=ctx.turn_id,
                    generation_id=ctx.generation_id,
                    event_type=event_type,
                    text=text,
                    payload=payload or {},
                )
            )
        except Exception:
            logger.warning("conversation audit append failed", exc_info=True)

    _envelope_seq = 0

    async def _send_envelope(
        self,
        ctx: TurnContext,
        *,
        kind: EventKind,
        payload: dict,
        seq: int | None = None,
        pts_ms: int | None = None,
    ) -> None:
        if self._room is None:
            return
        self._envelope_seq += 1
        envelope = StreamEnvelope(
            session_id=ctx.session_id,
            turn_id=ctx.turn_id,
            generation_id=ctx.generation_id,
            branch_state=self.session.branch_state,
            seq=seq if seq is not None else self._envelope_seq,
            kind=kind,
            pts_ms=pts_ms,
            payload=payload,
        )
        try:
            await self._room.local_participant.publish_data(
                json.dumps(envelope.to_wire(), ensure_ascii=False).encode("utf-8"),
                reliable=True,
                topic=DATA_TOPIC,
            )
        except Exception:
            logger.exception("publish_data failed")
