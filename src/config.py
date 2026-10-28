"""
src/config.py
Centralized application settings and observability callbacks.
"""

import os
from typing import Optional, Any
from dotenv import load_dotenv

# Ensure .env is explicitly loaded
load_dotenv(override=True)

# Keep a single singleton instance
_handler_instance: Optional[Any] = None


def get_langfuse_handler() -> Optional[Any]:
    """
    Initializes and returns a singleton Langfuse CallbackHandler.
    """
    global _handler_instance
    if _handler_instance is not None:
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

            _handler_instance = CallbackHandler(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            return _handler_instance
        except Exception as e:
            print(f"[Observability Warning] Failed to initialize Langfuse handler: {e}")
            return None

    return None


def flush_langfuse():
    """Forces pending event batches to sync immediately to the cloud dashboard."""
    global _handler_instance
    if _handler_instance and hasattr(_handler_instance, "flush"):
        try:
            _handler_instance.flush()
        except Exception as e:
            print(f"[Observability Warning] Error flushing traces: {e}")