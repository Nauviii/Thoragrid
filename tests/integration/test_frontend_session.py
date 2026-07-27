"""Integration tests for opaque browser session keys, against live Redis.

Consistent with the rest of this suite, these hit the real service rather than a mock: the
TTL and key-expiry semantics being asserted are Redis behaviour, and a fake would only test
the fake. Requires REDIS_URL to point at a reachable instance.
"""

import sys
from pathlib import Path

import pytest
import redis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import settings
from core.memory.frontend_session import (
    create_session_key, resolve_session_key, revoke_session_key,
)


@pytest.fixture
def session_key():
    """Create a session key and guarantee it is revoked afterwards."""
    key = create_session_key("jwt-token-value", "doctor", "doctor")
    yield key
    revoke_session_key(key)


def test_resolve_returns_the_stored_session(session_key):
    """A freshly created key must resolve back to exactly what was stored."""
    assert resolve_session_key(session_key) == {
        "token": "jwt-token-value", "username": "doctor", "role": "doctor",
    }


def test_each_login_gets_a_distinct_key():
    """Two logins must never share a key, or signing out of one would end the other."""
    first = create_session_key("jwt", "doctor", "doctor")
    second = create_session_key("jwt", "doctor", "doctor")
    try:
        assert first != second
    finally:
        revoke_session_key(first)
        revoke_session_key(second)


def test_revoked_key_stops_resolving(session_key):
    """Sign-out must invalidate immediately, not wait for the token to expire."""
    revoke_session_key(session_key)
    assert resolve_session_key(session_key) is None


def test_unknown_key_resolves_to_none():
    """A forged or stale cookie must resolve to None rather than raising."""
    assert resolve_session_key("this-key-was-never-issued") is None


def test_ttl_never_outlives_the_token(session_key):
    """The key's lifetime is pinned to the token's, so a resumed session can't outlive it."""
    client = redis.from_url(settings.redis_url, decode_responses=True)
    ttl = client.ttl(f"frontend_session:{session_key}")
    assert 0 < ttl <= settings.access_token_expire_minutes * 60