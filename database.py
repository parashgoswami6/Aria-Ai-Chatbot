"""
database.py — SQLite Chat History Manager
Handles: storing messages, retrieving history, clearing sessions
"""

import sqlite3
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE MANAGER
# ══════════════════════════════════════════════════════════════════════════════
class DatabaseManager:
    """Manages all SQLite operations for chat history."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection with row_factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row    # Access columns by name
        conn.execute("PRAGMA journal_mode=WAL")    # Better concurrent performance
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        schema = """
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT    NOT NULL UNIQUE,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
            updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS messages (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT    NOT NULL,
            role         TEXT    NOT NULL CHECK(role IN ('user', 'bot')),
            content      TEXT    NOT NULL,
            intent       TEXT,
            confidence   REAL,
            timestamp    TEXT    NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id);

        CREATE INDEX IF NOT EXISTS idx_messages_timestamp
            ON messages(timestamp);
        """
        try:
            with self._get_connection() as conn:
                conn.executescript(schema)
            logger.info(f"Database initialized at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    # ── Session management ─────────────────────────────────────────────────
    def create_session(self, session_id: str) -> bool:
        """Create a new chat session."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)",
                    (session_id,)
                )
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            return False

    def get_all_sessions(self) -> List[Dict]:
        """Retrieve all sessions with message count."""
        try:
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT
                        s.session_id,
                        s.created_at,
                        s.updated_at,
                        COUNT(m.id) as message_count
                    FROM sessions s
                    LEFT JOIN messages m ON s.session_id = m.session_id
                    GROUP BY s.session_id
                    ORDER BY s.updated_at DESC
                    LIMIT 50
                """).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to get sessions: {e}")
            return []

    # ── Message operations ─────────────────────────────────────────────────
    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> bool:
        """Save a single message to the database."""
        try:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            with self._get_connection() as conn:
                # Ensure session exists
                conn.execute(
                    "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)",
                    (session_id,)
                )
                # Insert message
                conn.execute("""
                    INSERT INTO messages
                        (session_id, role, content, intent, confidence, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (session_id, role, content, intent, confidence, timestamp))

                # Update session's updated_at
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (timestamp, session_id)
                )
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to save message: {e}")
            return False

    def save_exchange(
        self,
        session_id: str,
        user_msg: str,
        bot_msg: str,
        intent: str,
        confidence: float
    ) -> bool:
        """Save a user→bot exchange in one call."""
        ok1 = self.save_message(session_id, "user", user_msg)
        ok2 = self.save_message(session_id, "bot",  bot_msg, intent, confidence)
        return ok1 and ok2

    def get_history(self, session_id: str, limit: int = 100) -> List[Dict]:
        """Retrieve chat history for a session, ordered by time."""
        try:
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT role, content, intent, confidence, timestamp
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                """, (session_id, limit)).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to get history for {session_id}: {e}")
            return []

    def get_all_history(self, limit: int = 200) -> List[Dict]:
        """Get all messages across all sessions (for admin panel)."""
        try:
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT session_id, role, content, intent, confidence, timestamp
                    FROM messages
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,)).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to get all history: {e}")
            return []

    def clear_session(self, session_id: str) -> bool:
        """Delete all messages for a session."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "DELETE FROM messages WHERE session_id = ?",
                    (session_id,)
                )
                conn.execute(
                    "DELETE FROM sessions WHERE session_id = ?",
                    (session_id,)
                )
            logger.info(f"Session {session_id} cleared")
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get overall statistics."""
        try:
            with self._get_connection() as conn:
                total_msg = conn.execute(
                    "SELECT COUNT(*) FROM messages"
                ).fetchone()[0]

                total_sessions = conn.execute(
                    "SELECT COUNT(*) FROM sessions"
                ).fetchone()[0]

                top_intents = conn.execute("""
                    SELECT intent, COUNT(*) as count
                    FROM messages
                    WHERE role = 'bot' AND intent IS NOT NULL
                    GROUP BY intent
                    ORDER BY count DESC
                    LIMIT 5
                """).fetchall()

            return {
                "total_messages": total_msg,
                "total_sessions": total_sessions,
                "top_intents": [dict(r) for r in top_intents]
            }
        except sqlite3.Error as e:
            logger.error(f"Failed to get stats: {e}")
            return {"total_messages": 0, "total_sessions": 0, "top_intents": []}


# ── Singleton ──────────────────────────────────────────────────────────────────
_db_instance: Optional[DatabaseManager] = None

def get_db() -> DatabaseManager:
    """Return shared DatabaseManager singleton."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance
