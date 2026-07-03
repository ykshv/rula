from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ru_local_avatar_agent.voice.audit import (
    ConversationAuditEvent,
    SQLiteConversationAuditStore,
)


class ConversationAuditStoreTest(unittest.TestCase):
    def test_append_and_list_session_events_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteConversationAuditStore(Path(temp_dir) / "conversation.sqlite3")

            store.append(
                ConversationAuditEvent(
                    session_id="s1",
                    turn_id=1,
                    generation_id=1,
                    event_type="user_final",
                    text="Как тебя зовут?",
                    payload={"state": {"avatar_name": "Даздраперма"}},
                    created_at=1.0,
                )
            )
            store.append(
                ConversationAuditEvent(
                    session_id="s1",
                    turn_id=1,
                    generation_id=1,
                    event_type="assistant_final",
                    text="Меня зовут Даздраперма.",
                    created_at=2.0,
                )
            )

            events = store.list_session("s1")

        self.assertEqual(
            [event["event_type"] for event in events],
            ["user_final", "assistant_final"],
        )
        self.assertEqual(events[0]["payload"], {"state": {"avatar_name": "Даздраперма"}})


if __name__ == "__main__":
    unittest.main()
