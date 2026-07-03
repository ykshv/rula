from __future__ import annotations

import unittest

from ru_local_avatar_agent.voice.barge import verify_barge_in_transcript


class BargeInVerificationTest(unittest.TestCase):
    def test_rejects_exact_assistant_echo(self) -> None:
        result = verify_barge_in_transcript(
            "Хорошо. Не буду.",
            "Хорошо, не буду.",
        )

        self.assertFalse(result.confirmed)
        self.assertEqual(result.reason, "assistant_echo_substring")

    def test_rejects_partial_assistant_echo(self) -> None:
        result = verify_barge_in_transcript(
            "Хорошо",
            "Хорошо, не буду.",
        )

        self.assertFalse(result.confirmed)
        self.assertEqual(result.reason, "assistant_echo_substring")

    def test_accepts_emergency_interrupt(self) -> None:
        result = verify_barge_in_transcript(
            "Стоп, хватит",
            "Я сейчас объясню подробнее.",
        )

        self.assertTrue(result.confirmed)
        self.assertEqual(result.reason, "emergency_command")

    def test_accepts_new_user_speech(self) -> None:
        result = verify_barge_in_transcript(
            "Нет, расскажи про другое",
            "Хорошо, не буду.",
        )

        self.assertTrue(result.confirmed)
        self.assertEqual(result.reason, "new_user_speech")

    def test_rejects_empty_transcript(self) -> None:
        result = verify_barge_in_transcript("", "Хорошо, не буду.")

        self.assertFalse(result.confirmed)
        self.assertEqual(result.reason, "empty_transcript")


if __name__ == "__main__":
    unittest.main()
