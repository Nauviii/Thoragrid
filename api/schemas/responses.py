"""Pydantic response models for API endpoints."""

from datetime import datetime

from pydantic import BaseModel


class TokenResponse(BaseModel):
    """Response body for a successful login."""
    access_token: str
    token_type: str = "bearer"


class GradCAMFindingOut(BaseModel):
    """One condition's GradCAM heatmap and zone activation summary."""
    condition: str
    heatmap_url: str
    dominant_zones: list[str]
    aligned: bool


class LLMConditionOut(BaseModel):
    """One condition's clinical explanation from LLM Call 2."""
    name: str
    explanation: str
    dominant_zones: list[str]


class ImageAnalysisResponse(BaseModel):
    """Full response body for POST /analyze/xray."""
    interaction_id: str
    conversation_id: str
    all_scores: dict[str, float]
    above_threshold: list[str]
    low_confidence_flag: bool
    gradcam_findings: list[GradCAMFindingOut]
    conditions: list[LLMConditionOut]
    clinical_summary: str
    cross_specialty_notes: str | None
    latency_ms: int


class TextQAResponse(BaseModel):
    """Response body for POST /query."""
    interaction_id: str
    conversation_id: str
    answer: str
    cross_specialty_notes: str | None
    latency_ms: int


class FeedbackResponse(BaseModel):
    """Response body for POST /feedback."""
    id: str
    interaction_id: str
    is_correct: bool
    comment: str | None


class HistoryItemOut(BaseModel):
    """Summary of one past interaction for the history list."""
    id: str
    conversation_id: str
    interaction_type: str
    timestamp: datetime
    raw_query: str | None
    above_threshold: list[str] | None
    latency_ms: int | None


class HistoryResponse(BaseModel):
    """Response body for GET /history."""
    items: list[HistoryItemOut]
    total: int


class ConversationImageFindingOut(BaseModel):
    """One condition's GradCAM heatmap within a transcript turn; heatmap_url is None if re-signing failed."""
    condition: str
    heatmap_url: str | None
    dominant_zones: list[str]
    aligned: bool


class ConversationTurnOut(BaseModel):
    """One turn in a conversation transcript."""
    interaction_id: str
    interaction_type: str
    timestamp: datetime
    query: str | None = None
    answer: str | None = None
    above_threshold: list[str] | None = None
    clinical_summary: str | None = None
    xray_url: str | None = None
    gradcam_findings: list[ConversationImageFindingOut] | None = None


class ConversationTranscriptResponse(BaseModel):
    """Response body for GET /conversation/{conversation_id}."""
    conversation_id: str
    turns: list[ConversationTurnOut]


class ConversationCloseResponse(BaseModel):
    """Response body for DELETE /conversation/{conversation_id}."""
    conversation_id: str
    closed: bool

class SqlAgentResponse(BaseModel):
    """Response body for POST /agent/query."""
    sql_executed: str | None
    explanation: str
    rows: list[dict]
    row_count: int
    latency_ms: int