from __future__ import annotations

import unittest

from ru_local_avatar_agent.voice.trace import TurnTrace


class TurnTraceTest(unittest.TestCase):
    def test_summary_reports_phase_latencies_and_cache_counts(self) -> None:
        trace = TurnTrace(
            session_id="s1",
            turn_id=2,
            generation_id=3,
            speech_end_wall=10.0,
            latency_tier="instant_cached",
            playback_policy="cached",
        )
        trace.mark("speculation_start", 9.9)
        trace.mark("eot_commit", 10.2)
        trace.mark("tts_first_chunk", 10.3)
        trace.mark("first_audio", 10.55)
        trace.mark("audio_done", 12.0)
        trace.observe_tts_unit(cache_hit=True)

        summary = trace.summary()

        self.assertEqual(summary["from_speech_end_ms"]["eot_commit"], 200)
        self.assertEqual(summary["from_speech_end_ms"]["first_audio"], 550)
        self.assertEqual(summary["phase_ms"]["commit_to_first_audio"], 350)
        self.assertEqual(summary["tts"]["cache_hits"], 1)
        self.assertEqual(summary["latency_tier"], "instant_cached")


if __name__ == "__main__":
    unittest.main()
