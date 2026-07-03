from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

from .events import StreamEnvelope


@dataclass(frozen=True, slots=True)
class AudioFrame:
    pcm16: bytes
    sample_rate: int
    pts_ms: int
    speaker: str


@dataclass(frozen=True, slots=True)
class TranscriptDelta:
    text: str
    is_final: bool
    confidence: float
    pts_ms: int
    words: list[dict[str, float | str]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EotDecision:
    should_commit: bool
    should_speculate: bool
    shift_probability: float
    hold_probability: float
    backchannel_probability: float
    reason: str


@dataclass(frozen=True, slots=True)
class TokenDelta:
    text: str
    is_final: bool = False


@dataclass(frozen=True, slots=True)
class AudioChunk:
    pcm16: bytes
    sample_rate: int
    pts_ms: int
    duration_ms: int


@dataclass(frozen=True, slots=True)
class BlendshapeFrame:
    pts_ms: int
    values: dict[str, float]
    emotion: str | None = None


class SpeechRecognizer(Protocol):
    async def stream(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[TranscriptDelta]:
        ...


class TurnDetector(Protocol):
    async def decide(
        self,
        *,
        transcript: TranscriptDelta | None,
        recent_audio: AudioFrame | None,
    ) -> EotDecision:
        ...


class DialogueModel(Protocol):
    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[TokenDelta]:
        ...


class SpeechSynthesizer(Protocol):
    async def stream(
        self,
        text_chunk: str,
        *,
        voice_profile_id: str,
        generation_id: int,
    ) -> AsyncIterator[AudioChunk]:
        ...


class FaceAnimator(Protocol):
    async def stream(self, audio: AsyncIterator[AudioChunk]) -> AsyncIterator[BlendshapeFrame]:
        ...


class AvatarTimeline(Protocol):
    async def emit(self, envelope: StreamEnvelope) -> None:
        ...
