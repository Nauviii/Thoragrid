"""Pydantic request models for API endpoints."""

from pydantic import BaseModel


class TextQARequest(BaseModel):
    """Request body for POST /query — a free-text clinical question, optionally continuing a conversation."""
    query: str
    conversation_id: str | None = None


class FeedbackRequest(BaseModel):
    """Request body for POST /feedback."""
    interaction_id: str
    is_correct: bool
    comment: str | None = None

class SqlAgentRequest(BaseModel):
    """Request body for POST /agent/query — a natural language analytics question."""
    question: str

class ResumeSessionRequest(BaseModel):
    """Request body for POST /auth/resume — an opaque browser session key."""
    session_key: str