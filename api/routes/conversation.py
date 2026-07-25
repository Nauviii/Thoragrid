"""GET /conversation/{id} — read full transcript from Postgres.
DELETE /conversation/{id} — explicitly close working memory (Redis only)."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session as DBSession

from config.settings import settings
from api.middleware.auth import require_role, TokenPayload
from api.schemas.responses import (
    ConversationTranscriptResponse, ConversationTurnOut, ConversationImageFindingOut,
    ConversationCloseResponse,
)
from scripts.db_session import get_db
from scripts.db_models import Interaction, Session as UserSession
from core.memory.session_memory import end_session
from core.memory.conversation_history import get_conversation_transcript
from core.storage.supabase_storage import create_signed_url, xray_path, heatmap_path

router = APIRouter()


def _resign(bucket: str, path: str) -> str | None:
    """Return a fresh signed URL, or None if the object can no longer be signed (e.g. deleted)."""
    try:
        return create_signed_url(bucket, path)
    except Exception:
        return None


def _check_ownership(conversation_id: str, user: TokenPayload, db: DBSession) -> Interaction:
    """Return the conversation's anchor interaction, or raise 404/403 if inaccessible."""
    anchor = db.query(Interaction).filter_by(id=conversation_id).first()
    if anchor is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")

    if user.role != "admin":
        owner = db.query(UserSession).filter_by(id=anchor.session_id).first()
        if owner is None or owner.user_id != user.sub:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Not your conversation")

    return anchor


@router.get("/conversation/{conversation_id}", response_model=ConversationTranscriptResponse)
def get_conversation(
    conversation_id: str,
    user: Annotated[TokenPayload, Depends(require_role("admin", "doctor"))],
    db: Annotated[DBSession, Depends(get_db)],
) -> ConversationTranscriptResponse:
    """Return the full transcript of a conversation; image/heatmap URLs are re-signed on every call
    so a case opened long after analysis never serves an expired (>1h) Supabase signed URL."""
    _check_ownership(conversation_id, user, db)
    transcript = get_conversation_transcript(conversation_id, db)

    turns = []
    for t in transcript:
        if t["interaction_type"] == "image":
            t["xray_url"] = _resign(settings.supabase_xray_bucket, xray_path(t.pop("image_hash")))
            t["gradcam_findings"] = [
                ConversationImageFindingOut(
                    condition=g["condition"], dominant_zones=g["dominant_zones"], aligned=g["aligned"],
                    heatmap_url=_resign(settings.supabase_gradcam_bucket,
                                         heatmap_path(t["interaction_id"], g["condition"])),
                )
                for g in t.pop("gradcam_conditions")
            ]
        else:
            t.pop("image_hash", None)
            t.pop("gradcam_conditions", None)
        turns.append(ConversationTurnOut(**t))

    return ConversationTranscriptResponse(conversation_id=conversation_id, turns=turns)


@router.delete("/conversation/{conversation_id}", response_model=ConversationCloseResponse)
def close_conversation(
    conversation_id: str,
    user: Annotated[TokenPayload, Depends(require_role("admin", "doctor"))],
    db: Annotated[DBSession, Depends(get_db)],
) -> ConversationCloseResponse:
    """Explicitly close a conversation's working memory; permanent Postgres history is unaffected."""
    _check_ownership(conversation_id, user, db)
    end_session(conversation_id)
    return ConversationCloseResponse(conversation_id=conversation_id, closed=True)