from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ru_local_avatar_agent.voice.brain import ConversationBrain
from ru_local_avatar_agent.voice.streaming_tts import TtsChunk
from ru_local_avatar_agent.voice.tts_cache import TtsUnitCache


class TtsResponsePolicyTest(unittest.TestCase):
    def test_direct_identity_plan_uses_cached_latency_tier(self) -> None:
        plan = ConversationBrain().plan_response([], "Как тебя зовут?")

        self.assertEqual(plan.mode, "direct")
        self.assertEqual(plan.latency_tier, "instant_cached")
        self.assertEqual(plan.cache_policy, "prefer_cached")
        self.assertEqual(plan.playback_policy, "cached")

    def test_tts_cache_round_trips_without_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = TtsUnitCache(
                root=Path(tmp),
                voice="Serena",
                generation_config={"do_sample": False},
                max_chars=96,
                enabled=True,
            )
            chunk = TtsChunk(
                pcm=np.array([0.0, 0.25, -0.25], dtype=np.float32),
                sample_rate=24_000,
                pts_ms=80,
                is_final=True,
            )

            cache.put("Меня зовут Даздраперма.", [chunk])
            restored = cache.get("  Меня   зовут Даздраперма. ")

            self.assertIsNotNone(restored)
            self.assertEqual(len(restored or []), 1)
            self.assertEqual((restored or [])[0].sample_rate, 24_000)
            np.testing.assert_allclose((restored or [])[0].pcm, chunk.pcm)


if __name__ == "__main__":
    unittest.main()
