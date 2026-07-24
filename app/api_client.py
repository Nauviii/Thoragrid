"""HTTP client for the MedAssist FastAPI backend.

All calls go through the same JWT the user obtained at login — the frontend never talks
to Postgres, Pinecone, Groq, or Supabase directly. Role scoping, guardrails, and audit
logging all live behind these endpoints.
"""

import os

import requests

BASE_URL = os.getenv("MEDASSIST_API_URL", "http://127.0.0.1:8000")
_TIMEOUT = 120  # image analysis runs CNN + GradCAM + 2 LLM calls; needs a generous ceiling


class ApiError(Exception):
    """Raised when the backend returns a non-2xx response."""


def _auth_headers(token: str) -> dict:
    """Build the bearer auth header."""
    return {"Authorization": f"Bearer {token}"}


def _handle(response: requests.Response) -> dict:
    """Return parsed JSON, or raise ApiError carrying the backend's detail message."""
    if response.ok:
        return response.json()
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    raise ApiError(f"{response.status_code}: {detail}")


def login(username: str, password: str) -> str:
    """Authenticate and return a JWT access token."""
    response = requests.post(
        f"{BASE_URL}/auth/token",
        data={"username": username, "password": password},
        timeout=_TIMEOUT,
    )
    return _handle(response)["access_token"]


def analyze_xray(token: str, file_bytes: bytes, filename: str, conversation_id: str | None = None) -> dict:
    """Upload a chest X-ray for analysis, optionally continuing an existing conversation."""
    response = requests.post(
        f"{BASE_URL}/analyze/xray",
        headers=_auth_headers(token),
        files={"file": (filename, file_bytes, "image/png")},
        data={"conversation_id": conversation_id or ""},
        timeout=_TIMEOUT,
    )
    return _handle(response)


def ask_question(token: str, question: str, conversation_id: str | None = None) -> dict:
    """Ask a clinical question, optionally as a follow-up within a conversation."""
    response = requests.post(
        f"{BASE_URL}/query",
        headers=_auth_headers(token),
        json={"query": question, "conversation_id": conversation_id},
        timeout=_TIMEOUT,
    )
    return _handle(response)


def get_history(token: str, limit: int = 50, offset: int = 0) -> dict:
    """Fetch the current user's interaction history."""
    response = requests.get(
        f"{BASE_URL}/history",
        headers=_auth_headers(token),
        params={"limit": limit, "offset": offset},
        timeout=_TIMEOUT,
    )
    return _handle(response)


def get_conversation(token: str, conversation_id: str) -> dict:
    """Fetch a full conversation transcript; readable even after the conversation is closed."""
    response = requests.get(
        f"{BASE_URL}/conversation/{conversation_id}",
        headers=_auth_headers(token),
        timeout=_TIMEOUT,
    )
    return _handle(response)


def close_conversation(token: str, conversation_id: str) -> dict:
    """Close a conversation's working memory; the permanent transcript is unaffected."""
    response = requests.delete(
        f"{BASE_URL}/conversation/{conversation_id}",
        headers=_auth_headers(token),
        timeout=_TIMEOUT,
    )
    return _handle(response)


def submit_feedback(token: str, interaction_id: str, is_correct: bool, comment: str | None = None) -> dict:
    """Record agree/disagree feedback on one interaction."""
    response = requests.post(
        f"{BASE_URL}/feedback",
        headers=_auth_headers(token),
        json={"interaction_id": interaction_id, "is_correct": is_correct, "comment": comment},
        timeout=_TIMEOUT,
    )
    return _handle(response)


def agent_query(token: str, question: str) -> dict:
    """Run a natural-language analytics question through the read-only SQL agent."""
    response = requests.post(
        f"{BASE_URL}/agent/query",
        headers=_auth_headers(token),
        json={"question": question},
        timeout=_TIMEOUT,
    )
    return _handle(response)