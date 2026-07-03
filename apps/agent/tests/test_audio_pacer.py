from __future__ import annotations

import unittest

from ru_local_avatar_agent.voice.audio_pacer import (
    bounded_playout_timeout_s,
    pacing_threshold_samples,
    should_wait_for_response_end_before_start,
)


class AudioPacerBudgetTest(unittest.TestCase):
    def test_active_playback_waits_only_for_one_frame(self) -> None:
        threshold = pacing_threshold_samples(
            started=True,
            rebuffering=False,
            ended=False,
            frame_samples=480,
            prebuffer_samples=5_280,
            rebuffer_samples=3_360,
        )

        self.assertEqual(threshold, 480)

    def test_rebuffering_waits_for_rebuffer_budget(self) -> None:
        threshold = pacing_threshold_samples(
            started=True,
            rebuffering=True,
            ended=False,
            frame_samples=480,
            prebuffer_samples=5_280,
            rebuffer_samples=3_360,
        )

        self.assertEqual(threshold, 3_360)

    def test_unit_boundary_can_release_before_full_prebuffer(self) -> None:
        threshold = pacing_threshold_samples(
            started=False,
            rebuffering=False,
            ended=False,
            frame_samples=480,
            prebuffer_samples=43_200,
            rebuffer_samples=28_800,
            unit_boundary_ready=True,
            unit_start_samples=14_400,
        )

        self.assertEqual(threshold, 14_400)

    def test_unit_boundary_never_releases_less_than_one_frame(self) -> None:
        threshold = pacing_threshold_samples(
            started=False,
            rebuffering=False,
            ended=False,
            frame_samples=480,
            prebuffer_samples=43_200,
            rebuffer_samples=28_800,
            unit_boundary_ready=True,
            unit_start_samples=120,
        )

        self.assertEqual(threshold, 480)

    def test_after_response_mode_waits_before_first_frame(self) -> None:
        self.assertTrue(
            should_wait_for_response_end_before_start(
                start_after_end=True,
                started=False,
                ended=False,
            )
        )

    def test_after_response_mode_releases_when_response_ended(self) -> None:
        self.assertFalse(
            should_wait_for_response_end_before_start(
                start_after_end=True,
                started=False,
                ended=True,
            )
        )

    def test_streaming_mode_does_not_wait_for_response_end(self) -> None:
        self.assertFalse(
            should_wait_for_response_end_before_start(
                start_after_end=False,
                started=False,
                ended=False,
            )
        )

    def test_timeout_tracks_remaining_audio_plus_tail(self) -> None:
        timeout = bounded_playout_timeout_s(
            expected_audio_ms=2_000,
            first_push_at=10.0,
            now=11.25,
            tail_ms=800,
        )

        self.assertAlmostEqual(timeout, 1.55)

    def test_timeout_is_short_when_audio_should_already_be_done(self) -> None:
        timeout = bounded_playout_timeout_s(
            expected_audio_ms=1_000,
            first_push_at=10.0,
            now=12.50,
            tail_ms=800,
        )

        self.assertAlmostEqual(timeout, 0.8)

    def test_timeout_has_minimum_budget_before_first_push(self) -> None:
        timeout = bounded_playout_timeout_s(
            expected_audio_ms=0,
            first_push_at=None,
            now=12.50,
            tail_ms=100,
        )

        self.assertAlmostEqual(timeout, 0.25)


if __name__ == "__main__":
    unittest.main()
