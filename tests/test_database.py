"""
tests/test_database.py
Unit tests for SQLite document registry and conversation thread persistence.
"""

from pathlib import Path
import pytest
from src import database


@pytest.fixture(autouse=True)
def use_temp_db(tmp_path, monkeypatch):
    """Overrides the database path to use a temporary SQLite file and disables cloud sync for testing."""
    temp_db = tmp_path / "test_checkpoints.db"
    monkeypatch.setattr(database, "DB_PATH", temp_db)
    monkeypatch.setattr(database, "SUPABASE_URL", None)  # Ensures offline test isolation
    database.init_db()
    yield temp_db


def test_init_db_creates_tables():
    """Ensures tables are correctly initialized."""
    with database.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN"
            " ('document_registry', 'conversation_threads')"
        )
        tables = [r[0] for r in cursor.fetchall()]
        assert "document_registry" in tables
        assert "conversation_threads" in tables


def test_register_and_get_documents():
    """Tests registering a new PDF document and retrieving registered records."""
    database.register_document(
        file_hash="abc123hash",
        filename="thesis.pdf",
        filepath="/data/raw_pdfs/thesis.pdf",
    )

    docs = database.get_registered_documents()
    assert len(docs) == 1
    assert docs[0]["file_hash"] == "abc123hash"
    assert docs[0]["filename"] == "thesis.pdf"


def test_thread_lifecycle():
    """Tests saving, retrieving, updating, and deleting conversation threads."""
    file_hash = "abc123hash"
    thread_id = "thread-uuid-001"
    database.register_document(file_hash, "thesis.pdf", "/path/thesis.pdf")

    messages = [{"role": "user", "content": "What is the core theorem?"}]

    # 1. Save Thread
    database.save_thread_messages(
        thread_id=thread_id,
        file_hash=file_hash,
        title="What is the core theorem?",
        messages=messages,
    )

    threads = database.get_document_threads(file_hash)
    assert len(threads) == 1
    assert threads[0]["thread_id"] == thread_id
    assert threads[0]["messages"][0]["content"] == "What is the core theorem?"

    # 2. Update Thread (ON CONFLICT)
    messages.append(
        {"role": "assistant", "content": "The core theorem states..."}
    )
    database.save_thread_messages(
        thread_id=thread_id,
        file_hash=file_hash,
        title="What is the core theorem?",
        messages=messages,
    )

    updated_threads = database.get_document_threads(file_hash)
    assert len(updated_threads[0]["messages"]) == 2

    # 3. Delete Thread
    database.delete_thread(thread_id)
    assert len(database.get_document_threads(file_hash)) == 0