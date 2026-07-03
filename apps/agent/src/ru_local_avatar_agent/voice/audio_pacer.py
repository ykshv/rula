"""Continuous audio feeder for the LiveKit source.

Single responsibility: accept PCM whenever synthesis produces it and push a
gap-free 20 ms frame stream into the LiveKit AudioSource. Everything else
(face data, envelopes, metrics) lives outside the audio hot path — reliable
data-channel sends interleaved with capture_frame were the main source of
mid-word stutter.

Buffering policy:
- playback starts once `prebuffer_ms` is queued (or the utterance ended),
- a completed first TTS unit can release playback earlier once
  `unit_start_min_ms` is queued, so short acknowledgements do not wait for the
  next sentence,
- on underrun the pacer pauses and resumes only after `rebuffer_ms` is
  queued again — one audible pause instead of 20 ms machine-gun gaps.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable

import numpy as np

from ru_local_avatar_agent.voice.audio import float32_to_int16

logger = logging.getLogger(__name__)

FRAME_MS = 20
PLAYOUT_TAIL_MS = 800
PACING_TAIL_MS = 2_000


def pacing_threshold_samples(
    *,
    started: bool,
    rebuffering: bool,
    ended: bool,
    frame_samples: int,
    prebuffer_samples: int,
    rebuffer_samples: int,
    unit_boundary_ready: bool = False,
    unit_start_samples: int = 0,
) -> int:
    """Return the amount of buffered audio required before pushing a frame."""
    if ended:
        return 1
    if not started:
        if unit_boundary_ready:
            return max(frame_samples, min(prebuffer_samples, unit_start_samples))
        return prebuffer_samples
    if rebuffering:
        return rebuffer_samples
    return frame_samples


def should_wait_for_response_end_before_start(
    *,
    start_after_end: bool,
    started: bool,
    ended: bool,
) -> bool:
    """Return whether playback must wait for the full response buffer."""
    return start_after_end and not started and not ended


def bounded_playout_timeout_s(
    *,
    expected_audio_ms: float,
    first_push_at: float | None,
    now: float,
    tail_ms: int = PLAYOUT_TAIL_MS,
) -> float:
    """Return a bounded wait budget for source playout.

    LiveKit's source-level playout wait is transport-facing. It is useful, but
    it must not hold the conversation turn state indefinitely: while the worker
    is still in ``speaking``, incoming mic audio is intentionally gated. The
    budget is therefore based on our own pushed audio timeline plus a small
    tail for source jitter.
    """
    if first_push_at is None:
        return max(0.25, tail_ms / 1000)
    elapsed_ms = max(0.0, (now - first_push_at) * 1000)
    remaining_ms = max(0.0, expected_audio_ms - elapsed_ms)
    return max(0.25, (remaining_ms + tail_ms) / 1000)


class AudioPacer:
    def __init__(
        self,
        source,
        sample_rate: int,
        *,
        prebuffer_ms: int = 220,
        rebuffer_ms: int = 140,
        unit_start_min_ms: int = 600,
        start_after_end: bool = False,
        on_first_frame: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._source = source
        self._sample_rate = sample_rate
        self._samples_per_frame = sample_rate * FRAME_MS // 1000
        self._prebuffer_samples = sample_rate * prebuffer_ms // 1000
        self._rebuffer_samples = sample_rate * rebuffer_ms // 1000
        self._unit_start_samples = sample_rate * unit_start_min_ms // 1000
        self._start_after_end = start_after_end
        self._on_first_frame = on_first_frame

        self._buffer = np.empty(0, dtype=np.int16)
        self._data_ready = asyncio.Event()
        self._ended = False
        self._stopped = False
        self._paused = False
        self._unit_boundary_ready = False
        self._pushed_samples = 0
        self._first_push_at: float | None = None
        self._underruns = 0
        self._task = asyncio.create_task(self._run())

    # ------------------------------------------------------------------ API

    def submit(self, pcm_f32: np.ndarray) -> None:
        """Queue synthesized PCM (float32 mono at the source sample rate)."""
        if self._stopped or pcm_f32.size == 0:
            return
        self._buffer = np.concatenate([self._buffer, float32_to_int16(pcm_f32)])
        self._data_ready.set()

    def end_of_response(self) -> None:
        """No more audio will arrive; flush whatever is buffered."""
        self._ended = True
        self._data_ready.set()

    def notify_unit_complete(self) -> None:
        """Allow first playback at a safe TTS unit boundary.

        This only matters before playback starts. After the first frame, normal
        pacing and rebuffer policy own the stream.
        """
        if self._stopped or self._first_push_at is not None:
            return
        self._unit_boundary_ready = True
        self._data_ready.set()

    def pause(self) -> None:
        """Freeze frame pushing without dropping buffered audio (duck path).

        Frames already handed to the LiveKit source keep playing; keeping the
        source queue short bounds that tail. `resume()` continues seamlessly.
        """
        self._paused = True

    def resume(self) -> None:
        self._paused = False
        self._data_ready.set()

    @property
    def paused(self) -> bool:
        return self._paused

    async def stop(self, *, flush_queue: bool = True) -> None:
        """Cancel pacing immediately (barge-in path)."""
        self._stopped = True
        self._buffer = np.empty(0, dtype=np.int16)
        self._data_ready.set()
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        if flush_queue:
            with contextlib.suppress(Exception):
                self._source.clear_queue()

    async def wait_played(self, *, expected_audio_ms: float | None = None) -> bool:
        """Wait until submitted audio has had a bounded chance to play out.

        Returns ``False`` when LiveKit/source playout exceeded the turn-release
        budget. In that case the queue is cleared so stale audio cannot leak
        into the next turn.
        """
        expected_ms = max(float(expected_audio_ms or 0.0), self.pushed_ms)
        pacing_timeout_s = max(1.0, (expected_ms + PACING_TAIL_MS) / 1000)
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=pacing_timeout_s)
        except TimeoutError:
            logger.warning(
                "audio pacer task timed out after %.0f ms "
                "(expected_audio_ms=%.0f pushed_ms=%.0f); releasing turn",
                pacing_timeout_s * 1000,
                expected_ms,
                self.pushed_ms,
            )
            self._stopped = True
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            with contextlib.suppress(Exception):
                self._source.clear_queue()
            return False

        expected_ms = max(expected_ms, self.pushed_ms)
        timeout_s = bounded_playout_timeout_s(
            expected_audio_ms=expected_ms,
            first_push_at=self._first_push_at,
            now=time.monotonic(),
        )
        with contextlib.suppress(Exception):
            try:
                await asyncio.wait_for(self._source.wait_for_playout(), timeout=timeout_s)
                return True
            except TimeoutError:
                logger.warning(
                    "audio source playout wait timed out after %.0f ms "
                    "(expected_audio_ms=%.0f pushed_ms=%.0f); releasing turn",
                    timeout_s * 1000,
                    expected_ms,
                    self.pushed_ms,
                )
                with contextlib.suppress(Exception):
                    self._source.clear_queue()
                return False
        return True

    @property
    def pushed_ms(self) -> float:
        return self._pushed_samples * 1000 / self._sample_rate

    @property
    def buffered_ms(self) -> float:
        return self._buffer.shape[0] * 1000 / self._sample_rate

    @property
    def first_push_at(self) -> float | None:
        return self._first_push_at

    @property
    def underruns(self) -> int:
        return self._underruns

    # ------------------------------------------------------------------ loop

    async def _run(self) -> None:
        from livekit import rtc

        started = False
        rebuffering = False
        while not self._stopped:
            if self._paused:
                self._data_ready.clear()
                # Re-check after clear: resume() may have landed between the
                # outer check and the clear (lost-wakeup race).
                if self._paused:
                    await self._data_ready.wait()
                continue
            if should_wait_for_response_end_before_start(
                start_after_end=self._start_after_end,
                started=started,
                ended=self._ended,
            ):
                self._data_ready.clear()
                await self._data_ready.wait()
                continue

            threshold = pacing_threshold_samples(
                started=started,
                rebuffering=rebuffering,
                ended=self._ended,
                frame_samples=self._samples_per_frame,
                prebuffer_samples=self._prebuffer_samples,
                rebuffer_samples=self._rebuffer_samples,
                unit_boundary_ready=self._unit_boundary_ready,
                unit_start_samples=self._unit_start_samples,
            )
            if self._buffer.shape[0] < threshold:
                if self._ended and self._buffer.shape[0] == 0:
                    return
                if (
                    started
                    and not rebuffering
                    and not self._ended
                    and self._buffer.shape[0] < self._samples_per_frame
                ):
                    self._underruns += 1
                    rebuffering = True
                    logger.debug("audio pacer rebuffering (underrun #%d)", self._underruns)
                    threshold = pacing_threshold_samples(
                        started=started,
                        rebuffering=rebuffering,
                        ended=self._ended,
                        frame_samples=self._samples_per_frame,
                        prebuffer_samples=self._prebuffer_samples,
                        rebuffer_samples=self._rebuffer_samples,
                        unit_boundary_ready=self._unit_boundary_ready,
                        unit_start_samples=self._unit_start_samples,
                    )
                if self._buffer.shape[0] < threshold:
                    self._data_ready.clear()
                    await self._data_ready.wait()
                    continue

            started = True
            rebuffering = False
            chunk = self._buffer[: self._samples_per_frame]
            self._buffer = self._buffer[self._samples_per_frame :]
            if chunk.shape[0] < self._samples_per_frame:
                chunk = np.pad(chunk, (0, self._samples_per_frame - chunk.shape[0]))
            frame = rtc.AudioFrame.create(self._sample_rate, 1, self._samples_per_frame)
            np.frombuffer(frame.data, dtype=np.int16)[:] = chunk
            if self._first_push_at is None:
                self._first_push_at = time.monotonic()
                if self._on_first_frame is not None:
                    await self._on_first_frame()
            # capture_frame blocks on the source's internal queue and is the
            # only await in the loop — pacing comes from the source itself.
            await self._source.capture_frame(frame)
            self._pushed_samples += self._samples_per_frame
