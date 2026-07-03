from __future__ import annotations

import unittest

from fastapi import HTTPException

from ru_local_avatar_agent.api.app import SessionStore


class SessionStoreTest(unittest.TestCase):
    def test_cancel_release_frees_voice_limit(self) -> None:
        store = SessionStore(max_voice_sessions=1, ttl_seconds=60)
        first = store.create(voice=True, enforce_voice_limit=True)

        with self.assertRaises(HTTPException):
            store.create(voice=True, enforce_voice_limit=True)

        store.release_voice(first.session_id)
        second = store.create(voice=True, enforce_voice_limit=True)

        self.assertNotEqual(first.session_id, second.session_id)


if __name__ == "__main__":
    unittest.main()
