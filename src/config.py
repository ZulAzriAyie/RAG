"""
src/config.py
Centralized application settings and observability callbacks.
"""

import atexit
import logging
import os
import re
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)

_handler_instance: Optional[Any] = None


def get_langfuse_handler(session_id: Optional[str] = None) -> Optional[Any]:
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
    raw_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # Sanitize LANGFUSE_HOST (remove markdown, quotes, and square brackets)
    clean_host = re.sub(r"[\[\]\(\)\'\"]", "", raw_host).strip()
    if "http" in clean_host and " " in clean_host:
        clean_host = clean_host.split()[0]
    
    if not clean_host.startswith("http"):
        clean_host = f"https://{clean_host}"

    # Set sanitized host back into os.environ for the Langfuse OpenTelemetry exporter
    os.environ["LANGFUSE_HOST"] = clean_host

    if public_key and secret_key and "your-public-key" not in public_key:
        try:
            try:
                from langfuse.callback import CallbackHandler
            except ImportError:
                from langfuse.langchain import CallbackHandler

            _handler_instance = CallbackHandler(
                public_key=public_key,
                secret_key=secret_key,
                host=clean_host,
            )

            if session_id and hasattr(_handler_instance, "session_id"):
                _handler_instance.session_id = session_id

            atexit.register(flush_langfuse)
            print("[Observability Info] Langfuse CallbackHandler initialized cleanly.")
            return _handler_instance
        except Exception as e:
            print(f"[Observability Warning] Failed to initialize Langfuse handler: {e}")
            return None

    return None


def flush_langfuse() -> None:
    global _handler_instance
    if _handler_instance and hasattr(_handler_instance, "flush"):
        try:
            _handler_instance.flush()
            print("[Observability Info] Langfuse trace buffer flushed successfully.")
        except Exception as e:
            print(f"[Observability Warning] Error flushing traces: {e}")