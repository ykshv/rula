from __future__ import annotations

import unittest

from ru_local_avatar_agent.voice.tts_units import split_tts_units


class TtsUnitsTest(unittest.TestCase):
    def test_splits_multi_sentence_clause(self) -> None:
        self.assertEqual(
            split_tts_units("Живу, спасибо. А ты как?"),
            ["Живу, спасибо.", "А ты как?"],
        )

    def test_keeps_single_sentence_clause(self) -> None:
        self.assertEqual(
            split_tts_units("Привет, как ты?"),
            ["Привет, как ты?"],
        )

    def test_splits_long_comma_unit(self) -> None:
        self.assertEqual(
            split_tts_units("Я Даздраперма, виртуальный ассистент."),
            ["Я Даздраперма, виртуальный ассистент."],
        )

    def test_splits_long_unit_on_words(self) -> None:
        self.assertEqual(
            split_tts_units(
                "Шесть миллионов семьсот пятнадцать тысяч шестьсот сорок два "
                "виртуальных разговора надо произнести без резкой склейки."
            ),
            [
                "Шесть миллионов семьсот пятнадцать тысяч шестьсот сорок два "
                "виртуальных разговора надо",
                "произнести без резкой склейки.",
            ],
        )

    def test_handles_many_short_sentences_without_phrase_rules(self) -> None:
        self.assertEqual(
            split_tts_units("Да. Хорошо. Проверю."),
            ["Да.", "Хорошо.", "Проверю."],
        )

    def test_normalizes_whitespace(self) -> None:
        self.assertEqual(
            split_tts_units("  Первый ответ.   Второй ответ.  "),
            ["Первый ответ.", "Второй ответ."],
        )

    def test_empty_text(self) -> None:
        self.assertEqual(split_tts_units("   "), [])


if __name__ == "__main__":
    unittest.main()
