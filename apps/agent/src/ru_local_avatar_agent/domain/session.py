from __future__ import annotations

from dataclasses import dataclass
from time import monotonic_ns

from .events import BranchState, StreamEnvelope, TurnContext


def _now_ms() -> int:
    return monotonic_ns() // 1_000_000


@dataclass(slots=True)
class Transition:
    previous: BranchState
    current: BranchState
    turn_id: int
    generation_id: int
    at_ms: int
    reason: str


class InvalidTransition(RuntimeError):
    pass


class SessionStateMachine:
    """Single source of truth for turn IDs, generation IDs, and stale chunk drops."""

    def __init__(self, session_id: str) -> None:
        if not session_id:
            raise ValueError("session_id is required")
        self.session_id = session_id
        self.turn_id = 0
        self.generation_id = 0
        self.branch_state = BranchState.LISTENING
        self.transitions: list[Transition] = [
            Transition(
                previous=BranchState.LISTENING,
                current=BranchState.LISTENING,
                turn_id=0,
                generation_id=0,
                at_ms=_now_ms(),
                reason="created",
            )
        ]

    def current_context(self) -> TurnContext:
        return TurnContext(
            session_id=self.session_id,
            turn_id=self.turn_id,
            generation_id=self.generation_id,
            branch_state=self.branch_state,
        )

    def start_speculative(self, reason: str) -> TurnContext:
        if self.branch_state not in {BranchState.LISTENING, BranchState.DISCARDED}:
            raise InvalidTransition(f"cannot speculate from {self.branch_state}")
        self.turn_id += 1
        self.generation_id += 1
        self._transition(BranchState.SPECULATIVE, reason)
        return self.current_context()

    def commit_eot(self, reason: str) -> TurnContext:
        if self.branch_state == BranchState.SPECULATIVE:
            self._transition(BranchState.COMMITTED, reason)
            return self.current_context()
        if self.branch_state == BranchState.LISTENING:
            self.turn_id += 1
            self.generation_id += 1
            self._transition(BranchState.COMMITTED, reason)
            return self.current_context()
        raise InvalidTransition(f"cannot commit EOT from {self.branch_state}")

    def start_speaking(self, reason: str = "tts_started") -> TurnContext:
        if self.branch_state != BranchState.COMMITTED:
            raise InvalidTransition(f"cannot speak from {self.branch_state}")
        self._transition(BranchState.SPEAKING, reason)
        return self.current_context()

    def discard_speculative(self, generation_id: int, reason: str) -> Transition:
        if self.branch_state != BranchState.SPECULATIVE:
            raise InvalidTransition(f"cannot discard from {self.branch_state}")
        if generation_id != self.generation_id:
            return self.transitions[-1]
        self._transition(BranchState.DISCARDED, reason)
        self.generation_id += 1
        self._transition(BranchState.LISTENING, "ready_after_discard")
        return self.transitions[-1]

    def interrupt(self, reason: str) -> Transition:
        previous_generation = self.generation_id
        self.generation_id += 1
        self._transition(BranchState.INTERRUPTED, reason)
        self._transition(BranchState.LISTENING, f"ready_after_interrupt:{previous_generation}")
        return self.transitions[-1]

    def finish_turn(self, reason: str) -> Transition:
        if self.branch_state not in {BranchState.COMMITTED, BranchState.SPEAKING}:
            raise InvalidTransition(f"cannot finish turn from {self.branch_state}")
        finished_generation = self.generation_id
        self._transition(BranchState.LISTENING, f"{reason}:{finished_generation}")
        return self.transitions[-1]

    def is_current(self, envelope: StreamEnvelope) -> bool:
        return (
            envelope.session_id == self.session_id
            and envelope.turn_id == self.turn_id
            and envelope.generation_id == self.generation_id
            and envelope.branch_state == self.branch_state
        )

    def accepts_generation(self, envelope: StreamEnvelope) -> bool:
        return (
            envelope.session_id == self.session_id
            and envelope.generation_id == self.generation_id
        )

    def _transition(self, new_state: BranchState, reason: str) -> None:
        previous = self.branch_state
        self.branch_state = new_state
        self.transitions.append(
            Transition(
                previous=previous,
                current=new_state,
                turn_id=self.turn_id,
                generation_id=self.generation_id,
                at_ms=_now_ms(),
                reason=reason,
            )
        )
