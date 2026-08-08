"""
Short-term memory (current conversation) + long-term memory (durable facts/preferences)
+ conversation summaries, all stored locally in SQLite. Nothing here is sent anywhere
by default -- app/router.py + gemini_provider.py decide what (if anything) leaves the
machine, and only the summary/relevant memory subset, never the raw table.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import app_data_dir

DB_PATH = app_data_dir() / "friend.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS long_term_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact TEXT NOT NULL,
    created_at REAL NOT NULL,
    source TEXT DEFAULT 'user'
);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    covers_up_to_message_id INTEGER NOT NULL,
    created_at REAL NOT NULL
);
"""


@dataclass
class LongTermMemoryItem:
    id: int
    fact: str
    created_at: float
    source: str


class MemoryStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # --- short-term (conversation) ---
    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        self._conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, time.time()),
        )
        self._conn.commit()

    def get_recent_messages(self, conversation_id: str, limit: int = 20) -> list[dict]:
        cur = self._conn.execute(
            "SELECT id, role, content, created_at FROM messages WHERE conversation_id=? "
            "ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        )
        rows = cur.fetchall()
        rows.reverse()
        return [{"id": r[0], "role": r[1], "content": r[2], "created_at": r[3]} for r in rows]

    def clear_conversation(self, conversation_id: str) -> None:
        self._conn.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
        self._conn.commit()

    # --- long-term memory ---
    def add_long_term_fact(self, fact: str, source: str = "user") -> int:
        cur = self._conn.execute(
            "INSERT INTO long_term_memory (fact, created_at, source) VALUES (?, ?, ?)",
            (fact, time.time(), source),
        )
        self._conn.commit()
        return cur.lastrowid

    def list_long_term_facts(self) -> list[LongTermMemoryItem]:
        cur = self._conn.execute("SELECT id, fact, created_at, source FROM long_term_memory ORDER BY id")
        return [LongTermMemoryItem(*row) for row in cur.fetchall()]

    def delete_long_term_fact(self, fact_id: int) -> None:
        self._conn.execute("DELETE FROM long_term_memory WHERE id=?", (fact_id,))
        self._conn.commit()

    def clear_all_memory(self) -> None:
        self._conn.execute("DELETE FROM long_term_memory")
        self._conn.execute("DELETE FROM summaries")
        self._conn.commit()

    # --- summaries ---
    def save_summary(self, conversation_id: str, summary_text: str, covers_up_to_message_id: int) -> None:
        self._conn.execute(
            "INSERT INTO summaries (conversation_id, summary_text, covers_up_to_message_id, created_at) "
            "VALUES (?, ?, ?, ?)",
            (conversation_id, summary_text, covers_up_to_message_id, time.time()),
        )
        self._conn.commit()

    def get_latest_summary(self, conversation_id: str) -> str | None:
        cur = self._conn.execute(
            "SELECT summary_text FROM summaries WHERE conversation_id=? ORDER BY id DESC LIMIT 1",
            (conversation_id,),
        )
        row = cur.fetchone()
        return row[0] if row else None
