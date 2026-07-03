from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ru_local_avatar_agent.runtime import status


class RuntimeStatusTest(unittest.TestCase):
    def test_voice_avatar_is_not_ready_without_voice_pipeline(self) -> None:
        artifacts = {
            "stt": {"ok": True},
            "tts": {"ok": True},
            "a2f": {"ok": True},
            "llm": {"ok": True},
            "avatar": {"ok": True},
        }
        services = {
            "livekit_credentials": {"ok": True},
            "livekit_server": {"ok": True},
            "vllm": {"ok": True},
            "gpu": {"ok": True},
            "voice_pipeline": {"ok": False},
        }

        with (
            patch.object(status, "artifact_checks", return_value=artifacts),
            patch.object(status, "service_checks", return_value=services),
        ):
            payload = status.runtime_status(
                SimpleNamespace(
                    production=False,
                    profile_path="profiles/rtx5090.yaml",
                    voice_preset="Serena",
                    avatar_url="/assets/avatars/default-female/AliciaSolid_vrm-0.51.vrm",
                )
            )

        self.assertTrue(payload["ready"]["text_chat"])
        self.assertTrue(payload["ready"]["all_artifacts"])
        self.assertFalse(payload["ready"]["voice_avatar"])

    def test_voice_avatar_is_not_ready_without_gpu_headroom(self) -> None:
        artifacts = {
            "stt": {"ok": True},
            "tts": {"ok": True},
            "a2f": {"ok": True},
            "llm": {"ok": True},
            "avatar": {"ok": True},
        }
        services = {
            "livekit_credentials": {"ok": True},
            "livekit_server": {"ok": True},
            "vllm": {"ok": True},
            "gpu": {"ok": False, "detail": "insufficient_vram_headroom"},
            "voice_pipeline": {"ok": True},
        }

        with (
            patch.object(status, "artifact_checks", return_value=artifacts),
            patch.object(status, "service_checks", return_value=services),
        ):
            payload = status.runtime_status(
                SimpleNamespace(
                    production=False,
                    profile_path="profiles/rtx5090.yaml",
                    voice_preset="Serena",
                    avatar_url="/assets/avatars/default-female/AliciaSolid_vrm-0.51.vrm",
                )
            )

        self.assertTrue(payload["ready"]["text_chat"])
        self.assertFalse(payload["ready"]["voice_avatar"])


if __name__ == "__main__":
    unittest.main()
