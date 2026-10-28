"""
tests/test_auth.py
Unit tests for the authentication module.
"""

from src.auth import verify_credentials

# SHA-256 hash for 'admin123'
EXPECTED_HASH = "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9"


def test_verify_credentials_success():
    """Validates that entering the correct password returns True."""
    assert verify_credentials("admin123", EXPECTED_HASH) is True


def test_verify_credentials_failure():
    """Validates that entering an incorrect password returns False."""
    assert verify_credentials("wrongpassword", EXPECTED_HASH) is False
    assert verify_credentials("", EXPECTED_HASH) is False