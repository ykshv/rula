from __future__ import annotations

import unittest

from ru_local_avatar_agent.voice.brain import ConversationBrain


class ConversationOwnershipTest(unittest.TestCase):
    def test_assistant_question_stays_assistant_owned(self) -> None:
        brain = ConversationBrain()
        brain.record_user_turn("Тварь.", turn_id=1, generation_id=1)
        brain.record_assistant_final("Ты меня обидел. Почему так?", turn_id=1, generation_id=1)

        self.assertEqual(brain.state.pending_assistant_question, "Почему так?")

        brain.record_user_turn("Потому что", turn_id=2, generation_id=2)
        plan = brain.plan_response([], "Потому что")

        self.assertEqual(plan.mode, "direct")
        self.assertEqual(plan.intent, "assistant_question_answer")
        self.assertNotIn("Ты спросил", plan.direct_text)
        self.assertFalse(brain.state.pending_assistant_question)

    def test_ownership_correction_acknowledges_assistant_question(self) -> None:
        brain = ConversationBrain()
        brain.record_user_turn("Тварь.", turn_id=1, generation_id=1)
        brain.record_assistant_final("Ты меня обидел. Почему так?", turn_id=1, generation_id=1)
        brain.record_user_turn("Это неправда, это ты спросила.", turn_id=2, generation_id=2)

        plan = brain.plan_response([], "Это неправда, это ты спросила.")

        self.assertEqual(plan.mode, "direct")
        self.assertEqual(plan.intent, "ownership_correction")
        self.assertIn("это я спросила", plan.direct_text.casefold())
        self.assertIn("почему так", plan.direct_text.casefold())
        self.assertNotIn("Ты спросил", plan.direct_text)

    def test_assistant_memory_recall_uses_assistant_turns(self) -> None:
        brain = ConversationBrain()
        brain.record_user_turn("Привет", turn_id=1, generation_id=1)
        brain.record_assistant_final("Привет. Как ты?", turn_id=1, generation_id=1)

        plan = brain.plan_response([], "А что ты спросила?")

        self.assertEqual(plan.mode, "direct")
        self.assertIn("Как ты?", plan.direct_text)
        self.assertNotIn("Ты только что", plan.direct_text)


if __name__ == "__main__":
    unittest.main()
