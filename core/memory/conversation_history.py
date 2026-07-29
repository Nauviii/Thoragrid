"""Read conversation transcripts from Postgres for continuity and audit."""

from sqlalchemy.orm import Session
from scripts.db_models import Interaction


def get_conversation_transcript(
    conversation_id: str,
    db: Session,
    max_turns: int | None = None,
) -> list[dict]:
    """Return all interactions in a conversation, chronologically ordered, with their answers."""
    interactions = (
        db.query(Interaction)
        .filter_by(conversation_id=conversation_id)
        .order_by(Interaction.timestamp.asc())
        .all()
    )
    if max_turns:
        interactions = interactions[-max_turns:]

    turns = []
    for i in interactions:
        turn = {
            "interaction_id": i.id,
            "interaction_type": i.interaction_type,
            "timestamp": i.timestamp,
            "query": None, "answer": None,
            "above_threshold": None, "clinical_summary": None,
            "image_hash": None, "gradcam_conditions": None,
        }
        if i.interaction_type == "text":
            turn["query"] = i.raw_query
            if i.llm_output and i.llm_output.text_response:
                turn["answer"] = i.llm_output.text_response.get("answer")
        else:
            turn["above_threshold"] = i.cnn_result.above_threshold if i.cnn_result else []
            if i.llm_output and i.llm_output.call2_output:
                turn["clinical_summary"] = i.llm_output.call2_output.get("clinical_summary")

            turn["image_hash"] = i.image_hash
            turn["gradcam_conditions"] = [
                {"condition": g.condition, "dominant_zones": g.dominant_zones, "aligned": g.aligned}
                for g in i.gradcam_findings
            ]
        turns.append(turn)
    return turns


def build_prior_context_from_transcript(transcript: list[dict]) -> dict | None:
    """Derive a session-memory-shaped prior context (findings + conversation log) from a transcript."""
    if not transcript:
        return None

    above_threshold: list[str] = []
    for turn in reversed(transcript):
        if turn["interaction_type"] == "image":
            above_threshold = turn["above_threshold"] or []
            break

    conversation = []
    for turn in transcript:
        if turn["interaction_type"] == "text":
            if turn["query"]:
                conversation.append({"role": "user", "content": turn["query"]})
            if turn["answer"]:
                conversation.append({"role": "assistant", "content": turn["answer"]})
        else:
            findings = ", ".join(turn["above_threshold"] or []) or "no findings"
            conversation.append({
                "role": "system",
                "content": f"[Image analysis: {findings}] {turn['clinical_summary'] or ''}",
            })

    return {"above_threshold": above_threshold, "conversation": conversation}