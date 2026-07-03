from __future__ import annotations

import unittest

from ru_local_avatar_agent.voice.streaming_tts import resolve_generation_settings


class StreamingTtsConfigTest(unittest.TestCase):
    def test_respects_greedy_profile_for_main_and_subtalker(self) -> None:
        settings = resolve_generation_settings(
            {
                "do_sample": False,
                "top_k": 0,
                "top_p": 1.0,
                "temperature": 0.8,
                "repetition_penalty": 1.02,
                "subtalker_dosample": False,
                "subtalker_top_k": 0,
                "subtalker_top_p": 1.0,
                "subtalker_temperature": 0.8,
                "max_new_tokens": 384,
            }
        )

        self.assertFalse(settings.do_sample)
        self.assertFalse(settings.subtalker_do_sample)
        self.assertEqual(settings.top_k, 0)
        self.assertEqual(settings.subtalker_top_k, 0)
        self.assertEqual(settings.max_frames, 384)

    def test_rejects_invalid_sampling_config(self) -> None:
        with self.assertRaises(ValueError):
            resolve_generation_settings({"top_p": 0})
        with self.assertRaises(ValueError):
            resolve_generation_settings({"temperature": 0})
        with self.assertRaises(ValueError):
            resolve_generation_settings({"max_new_tokens": 0})


if __name__ == "__main__":
    unittest.main()
