from __future__ import annotations

import unittest

from ru_local_avatar_agent.domain.events import BranchState, EventKind
from ru_local_avatar_agent.domain.session import InvalidTransition, SessionStateMachine


class SessionStateMachineTest(unittest.TestCase):
    def test_speculative_commit_preserves_generation(self) -> None:
        session = SessionStateMachine("s1")
        speculative = session.start_speculative("vap_shift")

        committed = session.commit_eot("semantic_confirmed")

        self.assertEqual(speculative.generation_id, committed.generation_id)
        self.assertEqual(committed.branch_state, BranchState.COMMITTED)
        self.assertEqual(session.turn_id, 1)

    def test_discard_speculative_invalidates_stale_generation(self) -> None:
        session = SessionStateMachine("s1")
        speculative = session.start_speculative("vap_shift")
        stale = speculative.envelope(seq=1, kind=EventKind.PARTIAL_TEXT)

        session.discard_speculative(speculative.generation_id, "user_continued")

        self.assertEqual(session.branch_state, BranchState.LISTENING)
        self.assertFalse(session.accepts_generation(stale))

    def test_interrupt_invalidates_in_flight_speaking_chunks(self) -> None:
        session = SessionStateMachine("s1")
        committed = session.commit_eot("final_transcript")
        speaking = session.start_speaking()
        stale_audio = speaking.envelope(seq=5, kind=EventKind.SPEECH_SEGMENT)

        session.interrupt("barge_in")

        self.assertEqual(session.branch_state, BranchState.LISTENING)
        self.assertNotEqual(session.generation_id, committed.generation_id)
        self.assertFalse(session.accepts_generation(stale_audio))

    def test_cannot_speak_before_commit(self) -> None:
        session = SessionStateMachine("s1")

        with self.assertRaises(InvalidTransition):
            session.start_speaking()


if __name__ == "__main__":
    unittest.main()
