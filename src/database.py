"""
src/database.py
Database interactions for document registration and conversation thread persistence.
Features local SQLite primary storage with automated, non-blocking Supabase cloud backup sync.
"""

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables for DATABASE_URL
load_dotenv(override=True)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.resolve()
DB_PATH = BASE_DIR / "data" / "checkpoints.db"

# Optional Supabase Connection String
SUPABASE_URL = os.getenv("DATABASE_URL")
if SUPABASE_URL and SUPABASE_URL.startswith("postgres://"):
    SUPABASE_URL = SUPABASE_URL.replace("postgres://", "postgresql://", 1)


def get_connection() -> sqlite3.Connection:
    """Returns a local SQLite database connection with WAL mode enabled to prevent shutdown corruption."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    """Initializes local SQLite tables and remote Supabase backup tables if configured."""
    # 1. Initialize Local SQLite Primary Tables
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

    # 2. Initialize Remote Supabase Backup Tables (Optional)
    if SUPABASE_URL:
        try:
            from sqlalchemy import Column, DateTime, String, Text, create_engine, func
            from sqlalchemy.orm import DeclarativeBase

            class Base(DeclarativeBase):
                pass

            class DocumentRegistryModel(Base):
                __tablename__ = "document_registry"
                file_hash = Column(String, primary_key=True)
                filename = Column(String, nullable=False)
                filepath = Column(String, nullable=False)
                uploaded_at = Column(String, nullable=False)

            class ConversationThreadModel(Base):
                __tablename__ = "conversation_threads"
                thread_id = Column(String, primary_key=True)
                file_hash = Column(String, nullable=False)
                title = Column(String, nullable=False)
                created_at = Column(String, nullable=False)
                messages_json = Column(Text, nullable=False)

            engine = create_engine(SUPABASE_URL, pool_pre_ping=True)
            Base.metadata.create_all(bind=engine)
            logger.info("Supabase cloud backup schema initialized successfully.")
        except Exception as e:
            logger.warning(
                f"[Backup Warning] Failed to initialize Supabase schema: {e}"
            )


# --- Async Workers for Non-Blocking Cloud Sync ---

def _sync_document_to_supabase(
    file_hash: str, filename: str, filepath: str, uploaded_at: str
) -> None:
    """Background worker to replicate registered documents to Supabase."""
    if not SUPABASE_URL:
        return
    try:
        from sqlalchemy import Column, String, create_engine
        from sqlalchemy.orm import DeclarativeBase, sessionmaker

        class Base(DeclarativeBase):
            pass

        class DocumentRegistryModel(Base):
            __tablename__ = "document_registry"
            file_hash = Column(String, primary_key=True)
            filename = Column(String, nullable=False)
            filepath = Column(String, nullable=False)
            uploaded_at = Column(String, nullable=False)

        engine = create_engine(SUPABASE_URL, pool_pre_ping=True)
        Session = sessionmaker(bind=engine)
        session = Session()

        record = (
            session.query(DocumentRegistryModel)
            .filter(DocumentRegistryModel.file_hash == file_hash)
            .first()
        )
        if not record:
            record = DocumentRegistryModel(
                file_hash=file_hash,
                filename=filename,
                filepath=filepath,
                uploaded_at=uploaded_at,
            )
            session.add(record)
            session.commit()
        session.close()
    except Exception as e:
        logger.warning(f"[Backup Warning] Could not sync document to Supabase: {e}")


def _sync_thread_to_supabase(
    thread_id: str, file_hash: str, title: str, created_at: str, messages_json: str
) -> None:
    """Background worker to replicate conversation threads to Supabase."""
    if not SUPABASE_URL:
        return
    try:
        from sqlalchemy import Column, String, Text, create_engine
        from sqlalchemy.orm import DeclarativeBase, sessionmaker

        class Base(DeclarativeBase):
            pass

        class ConversationThreadModel(Base):
            __tablename__ = "conversation_threads"
            thread_id = Column(String, primary_key=True)
            file_hash = Column(String, nullable=False)
            title = Column(String, nullable=False)
            created_at = Column(String, nullable=False)
            messages_json = Column(Text, nullable=False)

        engine = create_engine(SUPABASE_URL, pool_pre_ping=True)
        Session = sessionmaker(bind=engine)
        session = Session()

        record = (
            session.query(ConversationThreadModel)
            .filter(ConversationThreadModel.thread_id == thread_id)
            .first()
        )
        if record:
            record.title = title
            record.messages_json = messages_json
        else:
            record = ConversationThreadModel(
                thread_id=thread_id,
                file_hash=file_hash,
                title=title,
                created_at=created_at,
                messages_json=messages_json,
            )
            session.add(record)

        session.commit()
        session.close()
    except Exception as e:
        logger.warning(f"[Backup Warning] Could not sync thread to Supabase: {e}")


def _delete_thread_from_supabase(thread_id: str) -> None:
    """Background worker to delete threads from Supabase."""
    if not SUPABASE_URL:
        return
    try:
        from sqlalchemy import Column, String, Text, create_engine
        from sqlalchemy.orm import DeclarativeBase, sessionmaker

        class Base(DeclarativeBase):
            pass

        class ConversationThreadModel(Base):
            __tablename__ = "conversation_threads"
            thread_id = Column(String, primary_key=True)
            file_hash = Column(String, nullable=False)
            title = Column(String, nullable=False)
            created_at = Column(String, nullable=False)
            messages_json = Column(Text, nullable=False)

        engine = create_engine(SUPABASE_URL, pool_pre_ping=True)
        Session = sessionmaker(bind=engine)
        session = Session()

        session.query(ConversationThreadModel).filter(
            ConversationThreadModel.thread_id == thread_id
        ).delete()

        session.commit()
        session.close()
    except Exception as e:
        logger.warning(f"[Backup Warning] Could not delete thread from Supabase: {e}")


# --- Public Interface Functions ---

def register_document(file_hash: str, filename: str, filepath: str) -> None:
    uploaded_at = datetime.now().isoformat()
    
    # 1. Local SQLite Instant Write
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
                INSERT OR IGNORE INTO document_registry (file_hash, filename, filepath, uploaded_at)
                VALUES (?, ?, ?, ?)
            """,
            (file_hash, filename, filepath, uploaded_at),
        )
        conn.commit()

    # 2. Async Cloud Backup Sync
    if SUPABASE_URL:
        threading.Thread(
            target=_sync_document_to_supabase,
            args=(file_hash, filename, filepath, uploaded_at),
            daemon=True,
        ).start()


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
    created_at = datetime.now().isoformat()
    messages_json = json.dumps(messages)

    # 1. Local SQLite Instant Write
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
            (thread_id, file_hash, title, created_at, messages_json),
        )
        conn.commit()

    # 2. Async Cloud Backup Sync
    if SUPABASE_URL:
        threading.Thread(
            target=_sync_thread_to_supabase,
            args=(thread_id, file_hash, title, created_at, messages_json),
            daemon=True,
        ).start()


def delete_thread(thread_id: str) -> None:
    # 1. Local SQLite Instant Delete
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM conversation_threads WHERE thread_id = ?", (thread_id,)
        )
        conn.commit()

    # 2. Async Cloud Backup Deletion
    if SUPABASE_URL:
        threading.Thread(
            target=_delete_thread_from_supabase,
            args=(thread_id,),
            daemon=True,
        ).start()