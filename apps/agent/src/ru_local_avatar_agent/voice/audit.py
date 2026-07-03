from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ConversationAuditEvent:
    session_id: str
    turn_id: int
    generation_id: int
    event_type: str
    text: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class ConversationAuditStore(Protocol):
    def append(self, event: ConversationAuditEvent) -> None:
        ...

    def list_session(self, session_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        ...


class NullConversationAuditStore:
    def append(self, event: ConversationAuditEvent) -> None:
        del event

    def list_session(self, session_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        del session_id, limit
        return []


class SQLiteConversationAuditStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def append(self, event: ConversationAuditEvent) -> None:
        payload_json = json.dumps(event.payload, ensure_ascii=False, sort_keys=True)
        with self._lock:
            conn = sqlite3.connect(self._path)
            try:
                conn.execute(
                    """
                    INSERT INTO conversation_events (
                        session_id,
                        turn_id,
                        generation_id,
                        event_type,
                        text,
                        payload_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.session_id,
                        event.turn_id,
                        event.generation_id,
                        event.event_type,
                        event.text,
                        payload_json,
                        event.created_at,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def list_session(self, session_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        bounded_limit = min(max(int(limit), 1), 500)
        with self._lock:
            conn = sqlite3.connect(self._path)
            try:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT
                        session_id,
                        turn_id,
                        generation_id,
                        event_type,
                        text,
                        payload_json,
                        created_at
                    FROM conversation_events
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (session_id, bounded_limit),
                ).fetchall()
                raw_events = [dict(row) for row in rows]
            finally:
                conn.close()
        events: list[dict[str, Any]] = []
        for row in reversed(raw_events):
            event = dict(row)
            payload_raw = event.pop("payload_json") or "{}"
            try:
                event["payload"] = json.loads(payload_raw)
            except json.JSONDecodeError:
                event["payload"] = {}
            events.append(event)
        return events

    def _init_db(self) -> None:
        with self._lock:
            conn = sqlite3.connect(self._path)
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        turn_id INTEGER NOT NULL,
                        generation_id INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        text TEXT NOT NULL DEFAULT '',
                        payload_json TEXT NOT NULL DEFAULT '{}',
                        created_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversation_events_session_created
                    ON conversation_events (session_id, created_at)
                    """
                )
                conn.commit()
            finally:
                conn.close()


def event_to_dict(event: ConversationAuditEvent) -> dict[str, Any]:
    return asdict(event)
