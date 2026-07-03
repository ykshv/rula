from __future__ import annotations

import math
import unittest

from ru_local_avatar_agent.voice.face_timeline import FACE_LEAD_MS, face_emit_horizon_ms


class FaceTimelineTest(unittest.TestCase):
    def test_live_horizon_keeps_face_ahead_of_audio(self) -> None:
        self.assertEqual(
            face_emit_horizon_ms(pushed_ms=1_200, flushing=False),
            1_200 + FACE_LEAD_MS,
        )

    def test_flushing_horizon_drains_a2f_tail(self) -> None:
        self.assertTrue(math.isinf(face_emit_horizon_ms(pushed_ms=1_200, flushing=True)))


if __name__ == "__main__":
    unittest.main()
