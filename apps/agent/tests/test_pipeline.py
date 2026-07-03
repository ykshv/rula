from __future__ import annotations

import unittest

from ru_local_avatar_agent.domain.contracts import TranscriptDelta
from ru_local_avatar_agent.domain.events import EventKind
from ru_local_avatar_agent.domain.session import SessionStateMachine
from ru_local_avatar_agent.runtime.mock_adapters import (
    MockDialogueModel,
    MockFaceAnimator,
    MockSpeechSynthesizer,
    ReactiveTurnDetector,
)
from ru_local_avatar_agent.runtime.pipeline import ConversationPipeline, InMemoryTimeline


class PipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_text_turn_emits_generation_scoped_events(self) -> None:
        timeline = InMemoryTimeline()
        session = SessionStateMachine("s1")
        pipeline = ConversationPipeline(
            turn_detector=ReactiveTurnDetector(),
            dialogue_model=MockDialogueModel(),
            synthesizer=MockSpeechSynthesizer(),
            face_animator=MockFaceAnimator(),
            timeline=timeline,
            token_target=4,
        )

        result = await pipeline.run_text_turn(
            state=session,
            transcript=TranscriptDelta(
                text="ok.",
                is_final=True,
                confidence=0.99,
                pts_ms=120,
            ),
        )

        self.assertTrue(result.committed)
        self.assertTrue(any(event.kind == EventKind.FINAL_TRANSCRIPT for event in result.envelopes))
        self.assertTrue(all(event.generation_id == 1 for event in result.envelopes))


if __name__ == "__main__":
    unittest.main()
