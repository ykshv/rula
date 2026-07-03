from __future__ import annotations

import unittest

from ru_local_avatar_agent.runtime.clause_chunker import ClauseChunker, normalize_tts_text


class ClauseChunkerTest(unittest.TestCase):
    def test_flushes_on_sentence_punctuation_after_speakable_unit(self) -> None:
        chunker = ClauseChunker(token_target=12)

        self.assertIsNone(chunker.push("Hello there."))
        chunk = chunker.push(" This is complete.")

        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.reason, "sentence_boundary")
        self.assertEqual(chunk.text, "Hello there. This is complete.")

    def test_does_not_flush_short_dangling_comma_clause(self) -> None:
        chunker = ClauseChunker(token_target=12)

        self.assertIsNone(chunker.push("Я — Даздраперма,"))
        chunk = chunker.push(" digital human с женским голосом.")

        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.reason, "sentence_boundary")
        self.assertEqual(chunk.text, "Я Даздраперма, digital human с женским голосом.")

    def test_finalizes_soft_boundary_for_tts(self) -> None:
        chunker = ClauseChunker(token_target=12)

        chunk = chunker.push("Слушай, я сейчас всё проверю,")

        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.reason, "clause_boundary")
        self.assertEqual(chunk.text, "Слушай, я сейчас всё проверю.")

    def test_token_target_does_not_flush_incomplete_russian_phrase(self) -> None:
        chunker = ClauseChunker(token_target=4)

        self.assertIsNone(chunker.push("Привет. "))
        for token in ["Чем ", "могу", " "]:
            self.assertIsNone(chunker.push(token))

        chunk = chunker.push("помочь?")
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.text, "Привет. Чем могу помочь?")

    def test_flushes_on_token_target_without_cutting_word(self) -> None:
        chunker = ClauseChunker(token_target=4)

        for token in [
            "conversation ",
            "quality ",
            "depends ",
            "on ",
            "stable ",
            "speech ",
        ]:
            self.assertIsNone(chunker.push(token))
        chunk = chunker.push("chunking")
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.reason, "token_target")
        self.assertEqual(chunk.text, "conversation quality depends on stable speech.")

        final = chunker.finish()
        self.assertIsNotNone(final)
        self.assertEqual(final.text, "chunking.")

    def test_flushes_internal_sentence_before_token_target(self) -> None:
        chunker = ClauseChunker(token_target=4)

        chunk = chunker.push("Привет! Как я")
        self.assertIsNone(chunk)

        self.assertIsNone(chunker.push(" могу"))
        chunk = chunker.push(" помочь?")
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.reason, "sentence_boundary")
        self.assertEqual(chunk.text, "Привет! Как я могу помочь?")

    def test_does_not_turn_sentence_fragment_into_tts_sentence(self) -> None:
        chunker = ClauseChunker(token_target=4)

        chunk = chunker.push("Поняла. Ты имеешь")
        self.assertIsNone(chunk)
        self.assertIsNone(chunker.push(" в виду"))
        self.assertIsNone(chunker.push(" длинную"))
        chunk = chunker.push(" реплику?")
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.text, "Поняла. Ты имеешь в виду длинную реплику?")

    def test_keeps_short_acknowledgement_with_followup(self) -> None:
        chunker = ClauseChunker(token_target=12)

        self.assertIsNone(chunker.push("Хорошо."))
        chunk = chunker.push(" Не буду.")

        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.reason, "sentence_boundary")
        self.assertEqual(chunk.text, "Хорошо. Не буду.")

    def test_preserves_subword_deltas_for_russian_tts(self) -> None:
        chunker = ClauseChunker(token_target=8)
        chunks = []

        for delta in ["В", "с", "ё", " отлично", ",", " сп", "ас", "и", "бо", "."]:
            chunk = chunker.push(delta)
            if chunk is not None:
                chunks.append(chunk.text)
        final = chunker.finish()
        if final is not None:
            chunks.append(final.text)

        self.assertEqual(" ".join(chunks), "Всё отлично, спасибо.")

    def test_normalizes_punctuation_spacing_before_tts(self) -> None:
        self.assertEqual(
            normalize_tts_text("Всё отлично , спасибо . А ты как ?"),
            "Всё отлично, спасибо. А ты как?",
        )

    def test_normalizes_em_dash_for_tts(self) -> None:
        self.assertEqual(normalize_tts_text("Я — Даздраперма,"), "Я Даздраперма,")


if __name__ == "__main__":
    unittest.main()
