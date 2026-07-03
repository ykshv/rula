from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from ru_local_avatar_agent.api.app import create_app


class ApiSecurityTest(unittest.TestCase):
    def test_production_disables_openapi_and_protects_admin_routes(self) -> None:
        env = {
            "RU_LOCAL_AVATAR_ENV": "production",
            "RU_LOCAL_AVATAR_TRUSTED_HOSTS": "testserver",
            "RU_LOCAL_AVATAR_ADMIN_TOKEN": "test-admin-token",
            "RU_LOCAL_AVATAR_VOICE": "0",
            "LIVEKIT_API_KEY": "devkey",
            "LIVEKIT_API_SECRET": "devsecret_change_me_local_only_32chars",
        }
        with patch.dict(os.environ, env, clear=False):
            client = TestClient(create_app())

        self.assertEqual(client.get("/openapi.json").status_code, 404)
        session_payload = client.post("/api/sessions").json()
        session_id = session_payload["session_id"]

        unauthenticated = client.post(f"/api/admin/sessions/{session_id}/cancel")
        self.assertEqual(unauthenticated.status_code, 404)

        authenticated = client.post(
            f"/api/admin/sessions/{session_id}/cancel",
            headers={"Authorization": "Bearer test-admin-token"},
        )
        self.assertEqual(authenticated.status_code, 200)

    def test_development_keeps_openapi_available(self) -> None:
        with patch.dict(os.environ, {"RU_LOCAL_AVATAR_ENV": "development"}, clear=False):
            client = TestClient(create_app())

        self.assertEqual(client.get("/openapi.json").status_code, 200)


if __name__ == "__main__":
    unittest.main()
