"""POST /query — text Q&A pipeline: guardrail check, retrieval, LLM call, DB logging."""

import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends
from pinecone import Index
from sqlalchemy.orm import Session as DBSession

from config.settings import settings
from api.dependencies import get_pinecone_index
from api.middleware.auth import require_role, TokenPayload
from api.schemas.requests import TextQARequest
from api.schemas.responses import TextQAResponse
from scripts.db_session import get_db
from scripts.db_models import Interaction, RAGLog, LLMOutput
from core.llm.orchestrator import run_text_llm_pipeline
from core.memory.session_memory import get_session, save_session, append_turn, is_conversation_closed
from core.memory.conversation_history import get_conversation_transcript, build_prior_context_from_transcript

router = APIRouter()


@router.post("/query", response_model=TextQAResponse)
def text_qa(
    body: TextQARequest,
    user: Annotated[TokenPayload, Depends(require_role("admin", "doctor"))],
    db: Annotated[DBSession, Depends(get_db)],
    index: Annotated[Index, Depends(get_pinecone_index)],
) -> TextQAResponse:
    """Run the text Q&A pipeline and persist the interaction, retrieval log, and LLM output.

    If body.conversation_id is given, this is a follow-up: working memory is fetched from Redis,
    or — if the TTL lapsed but the conversation was never explicitly closed — reconstructed from
    Postgres and revived. A conversation explicitly closed via DELETE /conversation/{id} is never
    auto-revived; it is treated as a standalone question instead.
    """
    start = time.perf_counter()

    interaction_id = str(uuid.uuid4())
    is_followup = bool(body.conversation_id)
    resolved_conversation_id = body.conversation_id or interaction_id

    prior_context = None
    if is_followup:
        prior_context = get_session(resolved_conversation_id)
        if prior_context is None and not is_conversation_closed(resolved_conversation_id):
            transcript = get_conversation_transcript(resolved_conversation_id, db, max_turns=6)
            prior_context = build_prior_context_from_transcript(transcript)
            if prior_context is not None:
                save_session(resolved_conversation_id, prior_context)

    interaction = Interaction(
        id=interaction_id, conversation_id=resolved_conversation_id,
        session_id=user.session_id, interaction_type="text", raw_query=body.query,
    )
    db.add(interaction)
    db.commit()
    db.refresh(interaction)

    bundle = run_text_llm_pipeline(
        body.query, index, settings.pinecone_namespace, prior_context=prior_context,
    )

    if bundle["rag_chunks"]:
        db.add(RAGLog(
            interaction_id=interaction.id, condition=None,
            query_used=bundle["query_used"],
            retrieved_ids=[c["chunk_id"] for c in bundle["rag_chunks"]],
            scores=[c["score"] for c in bundle["rag_chunks"]],
        ))
        db.commit()

    db.add(LLMOutput(
        interaction_id=interaction.id,
        call1_output=None,
        call2_output=None,
        text_response=bundle["answer_output"],
    ))

    interaction.latency_ms = int((time.perf_counter() - start) * 1000)
    db.commit()

    if not is_conversation_closed(resolved_conversation_id):
        append_turn(resolved_conversation_id, "user", body.query)
        append_turn(resolved_conversation_id, "assistant", bundle["answer_output"]["answer"])

    return TextQAResponse(
        interaction_id=interaction.id,
        conversation_id=resolved_conversation_id,
        answer=bundle["answer_output"]["answer"],
        latency_ms=interaction.latency_ms,
    )