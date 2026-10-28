"""
src/auth.py
Authentication module handling login state verification, credential hashing, and security barriers.
"""

import os
import hashlib
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def verify_credentials(password: str, expected_hash: str) -> bool:
    """Verifies an input password against the expected SHA-256 hash."""
    input_hash = hashlib.sha256(password.encode()).hexdigest()
    return input_hash == expected_hash


def render_login_barrier() -> bool:
    """
    Renders the authentication form if the user is unauthenticated.
    Returns True if authenticated, False if blocked.
    """
    if st.session_state.get("authenticated", False):
        return True

    st.title("🔒 Restricted Access — Thesis Agent")
    st.caption("Please log in to access research documents and LLM state machines.")

    with st.form("login_form", clear_on_submit=True):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Log In", use_container_width=True)

        if submit_button:
            expected_username = os.getenv("APP_USERNAME", "admin")
            expected_hash = os.getenv(
                "APP_PASSWORD_HASH",
                "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9"  # Default hash for 'admin123'
            )

            if username == expected_username and verify_credentials(password, expected_hash):
                st.session_state["authenticated"] = True
                st.success("Authentication successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    return False