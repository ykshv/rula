from __future__ import annotations

import unittest

from ru_local_avatar_agent.voice.text_match import speculative_transcript_matches


class SpeculativeTranscriptMatchTest(unittest.TestCase):
    def test_accepts_minor_russian_case_correction(self) -> None:
        self.assertTrue(
            speculative_transcript_matches(
                "Понял. Аниме девка.",
                "Понял. Аниме девку.",
            )
        )

    def test_accepts_small_tail_difference(self) -> None:
        self.assertTrue(
            speculative_transcript_matches(
                "Вот последние сессии не всё проговаривает",
                "Вот последние сессии не всё проговаривает блин",
            )
        )

    def test_rejects_short_ambiguous_turns(self) -> None:
        self.assertFalse(speculative_transcript_matches("да", "дам"))

    def test_rejects_negation_change(self) -> None:
        self.assertFalse(
            speculative_transcript_matches(
                "Можно запускать",
                "Нельзя запускать",
            )
        )

    def test_rejects_meaningful_different_turn(self) -> None:
        self.assertFalse(
            speculative_transcript_matches(
                "Расскажи о себе",
                "Выключи микрофон",
            )
        )


if __name__ == "__main__":
    unittest.main()
