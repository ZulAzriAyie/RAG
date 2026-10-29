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
    Registers an automated atexit hook to guarantee flushing on worker termination.
    
    Accepts an optional session_id to maintain backwards compatibility while
    allowing thread-level context enrichment.
    """
    global _handler_instance
    if _handler_instance is not None:
        # Update session_id if provided on subsequent calls
        if session_id and hasattr(_handler_instance, "session_id"):
            try:
                _handler_instance.session_id = session_id
            except Exception:
                pass
        return _handler_instance

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if public_key and secret_key and "your-public-key" not in public_key:
        try:
            try:
                from langfuse.callback import CallbackHandler
            except ImportError:
                from langfuse.langchain import CallbackHandler

            tags = ["production" if os.getenv("RENDER") else "development"]

            handler_kwargs = {
                "public_key": public_key,
                "secret_key": secret_key,
                "host": host,
                "tags": tags,
            }
            if session_id:
                handler_kwargs["session_id"] = session_id

            _handler_instance = CallbackHandler(**handler_kwargs)

            # Register automatic flush on Python process termination
            atexit.register(flush_langfuse)

            return _handler_instance
        except Exception as e:
            logger.warning(f"[Observability Warning] Failed to initialize Langfuse handler: {e}")
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