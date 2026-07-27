"""Opaque browser-session keys that map to an issued JWT, so a page refresh can resume a login.

Streamlit keeps no state across a browser reload, so something must survive on the client.
What survives here is an opaque random key, never the token: the JWT stays in Redis on the
server. A copied cookie is therefore worthless once the key is revoked, sign-out invalidates
immediately instead of waiting for the token to expire, and the token is never exposed to
JavaScript on the page.

The key's TTL is pinned to the token's own lifetime, so a resumed session can never outlive
the credential it stands for.
"""

import json
import secrets

import redis

from config.settings import settings

_r = redis.from_url(settings.redis_url, decode_responses=True)

_KEY_PREFIX = "frontend_session:"
_KEY_BYTES = 32  # 256 bits of entropy; far beyond guessability for a 30-minute window


def _redis_key(session_key: str) -> str:
    """Build the Redis key for an opaque browser session key."""
    return f"{_KEY_PREFIX}{session_key}"


def create_session_key(token: str, username: str, role: str) -> str:
    """Store an issued token under a fresh opaque key and return that key."""
    session_key = secrets.token_urlsafe(_KEY_BYTES)
    payload = json.dumps({"token": token, "username": username, "role": role})
    _r.set(
        _redis_key(session_key),
        payload,
        ex=settings.access_token_expire_minutes * 60,
    )
    return session_key


def resolve_session_key(session_key: str) -> dict | None:
    """Return the stored session for an opaque key, or None if unknown, revoked, or expired."""
    raw = _r.get(_redis_key(session_key))
    return json.loads(raw) if raw else None


def revoke_session_key(session_key: str) -> None:
    """Delete an opaque key so a copied cookie stops working immediately."""
    _r.delete(_redis_key(session_key))