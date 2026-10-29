"""
src/config.py
Centralized application settings and observability callbacks.
"""

import atexit
import logging
import os
from typing import Any, Optional
from dotenv import load_dotenv

# Ensure .env is explicitly loaded
load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Singleton reference for the callback handler
_handler_instance: Optional[Any] = None


def get_langfuse_handler(session_id: Optional[str] = None) -> Optional[Any]:
    """
    Initializes and returns a singleton Langfuse CallbackHandler.
    Automatically reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST
    from environment variables.
    """
    global _handler_instance
    if _handler_instance is not None:
        if session_id and hasattr(_handler_instance, "session_id"):
            try:
                _handler_instance.session_id = session_id
            except Exception:
                pass
        return _handler_instance

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if public_key and secret_key and "your-public-key" not in public_key:
        try:
            # Import compatibility for newer langfuse SDK versions
            try:
                from langfuse.callback import CallbackHandler
            except ImportError:
                from langfuse.langchain import CallbackHandler

            # Instantiate without passing secret_key explicitly
            # SDK reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST from env
            _handler_instance = CallbackHandler()

            if session_id and hasattr(_handler_instance, "session_id"):
                _handler_instance.session_id = session_id

            # Register automatic flush on Python process termination
            atexit.register(flush_langfuse)

            print("[Observability Info] Langfuse CallbackHandler initialized successfully.")
            return _handler_instance
        except Exception as e:
            print(f"[Observability Warning] Failed to initialize Langfuse handler: {e}")
            return None

    return None


def flush_langfuse() -> None:
    """Forces pending event batches to sync immediately to the cloud dashboard."""
    global _handler_instance
    if _handler_instance and hasattr(_handler_instance, "flush"):
        try:
            _handler_instance.flush()
            print("[Observability Info] Langfuse trace buffer flushed successfully.")
        except Exception as e:
            print(f"[Observability Warning] Error flushing traces: {e}")