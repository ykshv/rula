from __future__ import annotations

import unittest

from ru_local_avatar_agent.voice.brain import ConversationBrain


class CompoundQuestionTest(unittest.TestCase):
    def test_compound_question_queues_unanswered_questions(self) -> None:
        brain = ConversationBrain()
        brain.record_user_turn("Тварь.", turn_id=1, generation_id=1)
        text = "Как тебя зовут? Что вообще знаешь? С чего я начал наш разговор?"
        brain.record_user_turn(text, turn_id=2, generation_id=2)

        first = brain.plan_response([], text)

        self.assertEqual(first.mode, "direct")
        self.assertEqual(first.reason, "compound_question")
        self.assertEqual(first.direct_text, "Меня зовут Даздраперма.")
        self.assertEqual(
            brain.state.pending_user_questions,
            ["Что вообще знаешь?", "С чего я начал наш разговор?"],
        )

        brain.record_assistant_final(first.direct_text, turn_id=2, generation_id=2)
        brain.record_user_turn("А дальше?", turn_id=3, generation_id=3)
        second = brain.plan_response([], "А дальше?")

        self.assertEqual(second.mode, "direct")
        self.assertIn("голосом через аватар", second.direct_text)
        self.assertEqual(brain.state.pending_user_questions, ["С чего я начал наш разговор?"])

        brain.record_assistant_final(second.direct_text, turn_id=3, generation_id=3)
        brain.record_user_turn("А дальше?", turn_id=4, generation_id=4)
        third = brain.plan_response([], "А дальше?")

        self.assertEqual(third.mode, "direct")
        self.assertIn("Тварь.", third.direct_text)
        self.assertNotIn("Ты спросил, почему", third.direct_text)


if __name__ == "__main__":
    unittest.main()
