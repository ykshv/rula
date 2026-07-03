from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from ru_local_avatar_agent.domain.contracts import (
    AudioChunk,
    AudioFrame,
    BlendshapeFrame,
    DialogueModel,
    EotDecision,
    FaceAnimator,
    SpeechSynthesizer,
    TokenDelta,
    TranscriptDelta,
    TurnDetector,
)


class ReactiveTurnDetector:
    async def decide(
        self,
        *,
        transcript: TranscriptDelta | None,
        recent_audio: AudioFrame | None,
    ) -> EotDecision:
        text = transcript.text if transcript else ""
        final_signal = bool(transcript and transcript.is_final)
        shift = 0.78 if final_signal or text.endswith((".", "?", "!")) else 0.42
        return EotDecision(
            should_commit=shift >= 0.74,
            should_speculate=shift >= 0.60,
            shift_probability=shift,
            hold_probability=1.0 - shift,
            backchannel_probability=0.30 if recent_audio else 0.0,
            reason="reactive_final_or_boundary" if final_signal else "reactive_partial",
        )


class VapEvalTrackDetector:
    def __init__(self, shift_threshold: float = 0.68, backchannel_threshold: float = 0.72) -> None:
        self.shift_threshold = shift_threshold
        self.backchannel_threshold = backchannel_threshold

    async def decide(
        self,
        *,
        transcript: TranscriptDelta | None,
        recent_audio: AudioFrame | None,
    ) -> EotDecision:
        text = (transcript.text if transcript else "").lower()
        shift = 0.72 if any(marker in text for marker in ("ok", "done", "thanks")) else 0.51
        backchannel = 0.74 if recent_audio and not transcript else 0.20
        return EotDecision(
            should_commit=False,
            should_speculate=shift >= self.shift_threshold,
            shift_probability=shift,
            hold_probability=1.0 - shift,
            backchannel_probability=backchannel,
            reason="vap_eval_track",
        )


class MockDialogueModel(DialogueModel):
    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[TokenDelta]:
        del messages
        for token in ["Acknowledged,", " I", " will", " answer", " briefly.", " Ready."]:
            await asyncio.sleep(0)
            yield TokenDelta(text=token)
        yield TokenDelta(text="", is_final=True)


class MockSpeechSynthesizer(SpeechSynthesizer):
    async def stream(
        self,
        text_chunk: str,
        *,
        voice_profile_id: str,
        generation_id: int,
    ) -> AsyncIterator[AudioChunk]:
        del text_chunk, voice_profile_id, generation_id
        yield AudioChunk(pcm16=b"\x00" * 960, sample_rate=24_000, pts_ms=0, duration_ms=20)


class MockFaceAnimator(FaceAnimator):
    async def stream(self, audio: AsyncIterator[AudioChunk]) -> AsyncIterator[BlendshapeFrame]:
        async for chunk in audio:
            yield BlendshapeFrame(
                pts_ms=chunk.pts_ms,
                values={
                    "jawOpen": 0.18,
                    "mouthClose": 0.05,
                    "eyeBlinkLeft": 0.0,
                    "eyeBlinkRight": 0.0,
                },
                emotion="neutral",
            )
