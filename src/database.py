"""
src/database.py
Database interactions for document registration and conversation thread persistence.
"""

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List

BASE_DIR = Path(__file__).parent.parent.resolve()
DB_PATH = BASE_DIR / "data" / "checkpoints.db"


def get_connection() -> sqlite3.Connection:
  """Returns a database connection."""
  DB_PATH.parent.mkdir(parents=True, exist_ok=True)
  return sqlite3.connect(str(DB_PATH))


def init_db() -> None:
  """Initializes tables for tracking documents and conversation threads."""
  with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_registry (
                file_hash TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                uploaded_at TEXT NOT NULL
            )
        """)
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_threads (
                thread_id TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                messages_json TEXT NOT NULL,
                FOREIGN KEY (file_hash) REFERENCES document_registry (file_hash)
            )
        """)
    conn.commit()


def register_document(file_hash: str, filename: str, filepath: str) -> None:
  with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
        """
            INSERT OR IGNORE INTO document_registry (file_hash, filename, filepath, uploaded_at)
            VALUES (?, ?, ?, ?)
        """,
        (file_hash, filename, filepath, datetime.now().isoformat()),
    )
    conn.commit()


def get_registered_documents() -> List[Dict[str, str]]:
  with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT file_hash, filename, filepath FROM document_registry ORDER BY"
        " uploaded_at DESC"
    )
    rows = cursor.fetchall()
    return [
        {"file_hash": r[0], "filename": r[1], "filepath": r[2]} for r in rows
    ]


def get_document_threads(file_hash: str) -> List[Dict[str, Any]]:
  with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
        """
            SELECT thread_id, title, messages_json 
            FROM conversation_threads 
            WHERE file_hash = ? 
            ORDER BY created_at DESC
        """,
        (file_hash,),
    )
    rows = cursor.fetchall()
    return [{
        "thread_id": r[0],
        "title": r[1],
        "messages": json.loads(r[2]),
    } for r in rows]


def save_thread_messages(
    thread_id: str, file_hash: str, title: str, messages: list
) -> None:
  with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
        """
            INSERT INTO conversation_threads (thread_id, file_hash, title, created_at, messages_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                messages_json = excluded.messages_json,
                title = excluded.title
        """,
        (
            thread_id,
            file_hash,
            title,
            datetime.now().isoformat(),
            json.dumps(messages),
        ),
    )
    conn.commit()


def delete_thread(thread_id: str) -> None:
  with get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM conversation_threads WHERE thread_id = ?", (thread_id,)
    )
    conn.commit()