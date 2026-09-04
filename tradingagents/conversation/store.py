from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ConversationStore:
    """Small SQLite conversation store used by the FastAPI chat endpoint."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._setup()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _setup(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation_threads (
                    thread_id TEXT PRIMARY KEY,
                    current_ticker TEXT,
                    as_of_date TEXT,
                    last_intent TEXT,
                    research_context TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(thread_id) REFERENCES conversation_threads(thread_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_conversation_messages_thread_id
                ON conversation_messages(thread_id, id);
                """
            )

    def ensure_thread(
        self,
        thread_id: str | None = None,
        *,
        current_ticker: str | None = None,
        as_of_date: str | None = None,
    ) -> str:
        tid = (thread_id or uuid.uuid4().hex).strip()
        if not tid:
            tid = uuid.uuid4().hex
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_threads(
                    thread_id, current_ticker, as_of_date, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    current_ticker=COALESCE(excluded.current_ticker, current_ticker),
                    as_of_date=COALESCE(excluded.as_of_date, as_of_date),
                    updated_at=excluded.updated_at
                """,
                (tid, current_ticker, as_of_date, now, now),
            )
        return tid

    def append_message(self, thread_id: str, role: str, content: str) -> None:
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported conversation role: {role}")
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversation_messages(thread_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (thread_id, role, str(content), now),
            )
            conn.execute(
                "UPDATE conversation_threads SET updated_at=? WHERE thread_id=?",
                (now, thread_id),
            )

    def history(self, thread_id: str, *, limit: int = 12) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at FROM conversation_messages
                WHERE thread_id=? ORDER BY id DESC LIMIT ?
                """,
                (thread_id, max(1, int(limit))),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversation_threads WHERE thread_id=?",
                (thread_id,),
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        try:
            data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
        except json.JSONDecodeError:
            data["metadata"] = {}
        return data

    def update_context(
        self,
        thread_id: str,
        *,
        current_ticker: str | None = None,
        as_of_date: str | None = None,
        last_intent: str | None = None,
        research_context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        current = self.get_thread(thread_id) or {}
        old_metadata = current.get("metadata", {}) if isinstance(current.get("metadata"), dict) else {}
        if metadata:
            old_metadata.update(metadata)
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE conversation_threads SET
                    current_ticker=COALESCE(?, current_ticker),
                    as_of_date=COALESCE(?, as_of_date),
                    last_intent=COALESCE(?, last_intent),
                    research_context=COALESCE(?, research_context),
                    metadata_json=?,
                    updated_at=?
                WHERE thread_id=?
                """,
                (
                    current_ticker,
                    as_of_date,
                    last_intent,
                    research_context,
                    json.dumps(old_metadata, ensure_ascii=False),
                    now,
                    thread_id,
                ),
            )

    def reset(self, thread_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM conversation_threads WHERE thread_id=?", (thread_id,))
        return cur.rowcount > 0
