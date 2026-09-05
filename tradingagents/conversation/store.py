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
    """SQLite store for conversation memory and immutable research versions."""

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
                    FOREIGN KEY(thread_id)
                        REFERENCES conversation_threads(thread_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS research_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    parent_version_id INTEGER,
                    version_kind TEXT NOT NULL DEFAULT 'research',
                    payload_json TEXT NOT NULL,
                    audit_status TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(thread_id)
                        REFERENCES conversation_threads(thread_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY(parent_version_id)
                        REFERENCES research_versions(id)
                );

                CREATE INDEX IF NOT EXISTS idx_conversation_messages_thread_id
                ON conversation_messages(thread_id, id);

                CREATE INDEX IF NOT EXISTS idx_research_versions_thread
                ON research_versions(thread_id, id);
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
                """
                INSERT INTO conversation_messages(
                    thread_id, role, content, created_at
                ) VALUES (?, ?, ?, ?)
                """,
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
                SELECT role, content, created_at
                FROM conversation_messages
                WHERE thread_id=?
                ORDER BY id DESC LIMIT ?
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
        old_metadata = (
            current.get("metadata", {})
            if isinstance(current.get("metadata"), dict)
            else {}
        )
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

    def save_research_version(
        self,
        thread_id: str,
        payload: dict[str, Any],
        *,
        audit_status: str = "",
        version_kind: str = "research",
    ) -> int:
        """Append an immutable research version and make it active."""

        now = _utc_now()
        with self._connect() as conn:
            active = conn.execute(
                """
                SELECT id FROM research_versions
                WHERE thread_id=? AND is_active=1
                ORDER BY id DESC LIMIT 1
                """,
                (thread_id,),
            ).fetchone()
            parent_id = int(active["id"]) if active else None
            conn.execute(
                "UPDATE research_versions SET is_active=0 WHERE thread_id=?",
                (thread_id,),
            )
            cur = conn.execute(
                """
                INSERT INTO research_versions(
                    thread_id, parent_version_id, version_kind,
                    payload_json, audit_status, created_at, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    thread_id,
                    parent_id,
                    version_kind,
                    json.dumps(payload, ensure_ascii=False),
                    str(audit_status or ""),
                    now,
                ),
            )
            return int(cur.lastrowid)

    @staticmethod
    def _decode_version(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        data = dict(row)
        try:
            data["payload"] = json.loads(data.pop("payload_json") or "{}")
        except json.JSONDecodeError:
            data["payload"] = {}
        data["is_active"] = bool(data.get("is_active"))
        return data

    def get_active_research_version(
        self,
        thread_id: str,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM research_versions
                WHERE thread_id=? AND is_active=1
                ORDER BY id DESC LIMIT 1
                """,
                (thread_id,),
            ).fetchone()
        return self._decode_version(row)

    def list_research_versions(
        self,
        thread_id: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM research_versions
                WHERE thread_id=?
                ORDER BY id DESC LIMIT ?
                """,
                (thread_id, max(1, min(int(limit), 100))),
            ).fetchall()
        return [self._decode_version(row) or {} for row in rows]

    def rollback_research_version(
        self,
        thread_id: str,
        version_id: int | None = None,
    ) -> dict[str, Any] | None:
        """Activate a previous immutable version; never delete history."""

        with self._connect() as conn:
            active = conn.execute(
                """
                SELECT * FROM research_versions
                WHERE thread_id=? AND is_active=1
                ORDER BY id DESC LIMIT 1
                """,
                (thread_id,),
            ).fetchone()
            if active is None:
                return None

            target = None
            if version_id is not None:
                target = conn.execute(
                    """
                    SELECT * FROM research_versions
                    WHERE thread_id=? AND id=?
                    """,
                    (thread_id, int(version_id)),
                ).fetchone()
            else:
                parent_id = active["parent_version_id"]
                if parent_id is not None:
                    target = conn.execute(
                        """
                        SELECT * FROM research_versions
                        WHERE thread_id=? AND id=?
                        """,
                        (thread_id, int(parent_id)),
                    ).fetchone()

            if target is None:
                return None

            conn.execute(
                "UPDATE research_versions SET is_active=0 WHERE thread_id=?",
                (thread_id,),
            )
            conn.execute(
                "UPDATE research_versions SET is_active=1 WHERE id=?",
                (int(target["id"]),),
            )

            decoded = self._decode_version(target) or {}
            payload = decoded.get("payload", {})
            if isinstance(payload, dict):
                context = str(payload.get("research_context", "") or "")
                ticker = payload.get("ticker")
                as_of_date = payload.get("as_of_date")
                conn.execute(
                    """
                    UPDATE conversation_threads SET
                        research_context=?,
                        current_ticker=COALESCE(?, current_ticker),
                        as_of_date=COALESCE(?, as_of_date),
                        last_intent='rollback',
                        updated_at=?
                    WHERE thread_id=?
                    """,
                    (context, ticker, as_of_date, _utc_now(), thread_id),
                )
            return decoded

    def reset(self, thread_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM conversation_threads WHERE thread_id=?",
                (thread_id,),
            )
        return cur.rowcount > 0
