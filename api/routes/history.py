"""GET /history — paginated interaction history, scoped to the current doctor unless admin."""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DBSession

from api.middleware.auth import require_role, TokenPayload
from api.schemas.responses import HistoryItemOut, HistoryResponse
from scripts.db_session import get_db
from scripts.db_models import Interaction, Session as UserSession

router = APIRouter()


@router.get("/history", response_model=HistoryResponse)
def get_history(
    user: Annotated[TokenPayload, Depends(require_role("admin", "doctor"))],
    db: Annotated[DBSession, Depends(get_db)],
    limit: int = 20,
    offset: int = 0,
) -> HistoryResponse:
    """Return paginated interaction history; doctors see only their own sessions, admin sees all."""
    query = db.query(Interaction).join(UserSession, Interaction.session_id == UserSession.id)
    if user.role != "admin":
        query = query.filter(UserSession.user_id == user.sub)

    total = query.count()
    interactions = (
        query.order_by(Interaction.timestamp.desc()).offset(offset).limit(limit).all()
    )

    items = [
        HistoryItemOut(
            id=i.id, conversation_id=i.conversation_id, interaction_type=i.interaction_type, timestamp=i.timestamp,
            raw_query=i.raw_query,
            above_threshold=i.cnn_result.above_threshold if i.cnn_result else None,
            latency_ms=i.latency_ms,
            feedback=i.feedback.is_correct if i.feedback else None,
        )
        for i in interactions
    ]

    return HistoryResponse(items=items, total=total)