from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class BranchState(StrEnum):
    LISTENING = "listening"
    SPECULATIVE = "speculative"
    COMMITTED = "committed"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    DISCARDED = "discarded"


class EventKind(StrEnum):
    TURN_STATE = "turn.state"
    TURN_TRACE = "turn.trace"
    PARTIAL_TRANSCRIPT = "turn.partial_transcript"
    FINAL_TRANSCRIPT = "turn.final_transcript"
    PARTIAL_TEXT = "assistant.partial_text"
    SPEECH_SEGMENT = "assistant.speech_segment"
    AVATAR_STATE = "avatar.state"
    AVATAR_BLENDSHAPE_FRAME = "avatar.blendshape_frame"
    AVATAR_EMOTION = "avatar.emotion"
    AVATAR_GESTURE = "avatar.gesture"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class StreamEnvelope:
    session_id: str
    turn_id: int
    generation_id: int
    branch_state: BranchState
    seq: int
    kind: EventKind
    pts_ms: int | None = None
    emitted_at_ms: int = field(default_factory=lambda: time.time_ns() // 1_000_000)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        data = asdict(self)
        data["branch_state"] = self.branch_state.value
        data["kind"] = self.kind.value
        return data


@dataclass(frozen=True, slots=True)
class TurnContext:
    session_id: str
    turn_id: int
    generation_id: int
    branch_state: BranchState

    def envelope(
        self,
        *,
        seq: int,
        kind: EventKind,
        pts_ms: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> StreamEnvelope:
        return StreamEnvelope(
            session_id=self.session_id,
            turn_id=self.turn_id,
            generation_id=self.generation_id,
            branch_state=self.branch_state,
            seq=seq,
            kind=kind,
            pts_ms=pts_ms,
            payload=payload or {},
        )
