"""POST /analyze/xray — full image pipeline: CLIP validate, CNN, GradCAM, RAG+LLM, DB logging.

Route is a sync def deliberately: every SDK used here (Groq, Pinecone, supabase-py,
SQLAlchemy, PyTorch inference) is synchronous. FastAPI runs sync def routes in a
threadpool automatically, which is correct for blocking calls.
"""

import base64
import hashlib
import io
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from PIL import Image
from pinecone import Index
from sqlalchemy.orm import Session as DBSession
from torch.nn import Module as TorchModule

from config.settings import settings
from api.dependencies import get_cnn_model, get_pinecone_index
from api.middleware.auth import require_role, TokenPayload
from api.schemas.responses import ImageAnalysisResponse, GradCAMFindingOut, LLMConditionOut
from scripts.db_session import get_db
from scripts.db_models import Interaction, CNNResult, GradCAMFinding, RAGLog, LLMOutput
from core.clip.validator import validate as clip_validate
from core.cnn.inference import run_inference
from core.gradcam.explainer import run_gradcam
from core.llm.orchestrator import run_image_llm_pipeline
from core.storage.supabase_storage import upload_and_sign
from core.memory.session_memory import get_session, save_session, is_conversation_closed

router = APIRouter()


@router.post("/analyze/xray", response_model=ImageAnalysisResponse)
def analyze_xray(
    file: UploadFile,
    user: Annotated[TokenPayload, Depends(require_role("admin", "doctor"))],
    db: Annotated[DBSession, Depends(get_db)],
    model: Annotated[TorchModule, Depends(get_cnn_model)],
    index: Annotated[Index, Depends(get_pinecone_index)],
    conversation_id: Annotated[str | None, Form()] = None,
) -> ImageAnalysisResponse:
    """Validate, run CNN+GradCAM, retrieve+explain via LLM, and persist the full interaction.

    conversation_id, if given, anchors this upload to an existing conversation (e.g. a second
    view of the same case) — working memory is overwritten with the newest findings while the
    conversation log keeps growing. If omitted, this upload starts a brand new conversation.
    """
    start = time.perf_counter()

    raw_bytes = file.file.read()
    image = Image.open(io.BytesIO(raw_bytes))

    validation = clip_validate(image)
    if not validation.is_valid:
        # Structured detail: the UI renders reason-specific guidance from `code`, while
        # `reason` keeps the scores for logs and support without being shown to clinicians.
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": validation.code, "reason": validation.reason},
        )

    inference_out = run_inference(image, model)

    # Normalize to PNG regardless of the original upload format (JPEG, BMP, etc.) —
    # keeps the storage extension/content-type truthful and applies Pillow's lossless
    # PNG optimization. No lossy compression is applied: pixel values are unaltered,
    # which matters for diagnostic imagery that CNN/GradCAM analysis depends on.
    png_buffer = io.BytesIO()
    image.convert("RGB").save(png_buffer, format="PNG", optimize=True)
    normalized_bytes = png_buffer.getvalue()

    image_hash = hashlib.sha256(normalized_bytes).hexdigest()
    xray_url = upload_and_sign(
        settings.supabase_xray_bucket, f"{image_hash}.png", normalized_bytes, "image/png"
    )

    interaction_id = str(uuid.uuid4())
    resolved_conversation_id = conversation_id or interaction_id

    interaction = Interaction(
        id=interaction_id, conversation_id=resolved_conversation_id,
        session_id=user.session_id, interaction_type="image",
        image_hash=image_hash, xray_storage_url=xray_url,
    )
    db.add(interaction)
    db.commit()
    db.refresh(interaction)

    db.add(CNNResult(
        interaction_id=interaction.id,
        all_scores=inference_out["all_scores"],
        above_threshold=inference_out["above_threshold"],
        low_confidence_flag=inference_out["low_confidence_flag"],
    ))
    db.commit()

    gradcam_out = run_gradcam(image, model, inference_out["above_threshold"])

    gradcam_findings_out: list[GradCAMFindingOut] = []
    for condition, finding in gradcam_out["gradcam_results"].items():
        heatmap_bytes = base64.b64decode(finding["heatmap_b64"])
        heatmap_url = upload_and_sign(
            settings.supabase_gradcam_bucket,
            f"{interaction.id}/{condition}.png", heatmap_bytes, "image/png",
        )
        db.add(GradCAMFinding(
            interaction_id=interaction.id, condition=condition,
            heatmap_storage_url=heatmap_url,
            dominant_zones=finding["dominant_zones"],
            aligned=finding["aligned"], zone_stats=finding["zone_stats"],
        ))
        gradcam_findings_out.append(GradCAMFindingOut(
            condition=condition, heatmap_url=heatmap_url,
            dominant_zones=finding["dominant_zones"], aligned=finding["aligned"],
        ))
    db.commit()

    bundle = run_image_llm_pipeline(
        all_scores=inference_out["all_scores"],
        above_threshold=inference_out["above_threshold"],
        low_confidence_flag=inference_out["low_confidence_flag"],
        gradcam_results=gradcam_out["gradcam_results"],
        semantic_context=gradcam_out["semantic_context"],
        index=index, namespace=settings.pinecone_namespace,
    )

    if bundle["llm1_output"] is not None:
        chunks_by_condition: dict[str, list[dict]] = {}
        for chunk in bundle["rag_chunks"]:
            chunks_by_condition.setdefault(chunk["condition"], []).append(chunk)

        query_by_condition = {q["condition"]: q["query"] for q in bundle["llm1_output"]["rag_queries"]}

        for condition, chunks in chunks_by_condition.items():
            db.add(RAGLog(
                interaction_id=interaction.id, condition=condition,
                query_used=query_by_condition.get(condition, ""),
                retrieved_ids=[c["chunk_id"] for c in chunks],
                scores=[c["score"] for c in chunks],
            ))
        db.commit()

    db.add(LLMOutput(
        interaction_id=interaction.id,
        call1_output=bundle["llm1_output"],
        call2_output=bundle["llm2_output"],
        text_response=None,
    ))

    interaction.latency_ms = int((time.perf_counter() - start) * 1000)
    db.commit()

    # Update working memory: overwrite current findings, keep appending to the conversation log
    if not is_conversation_closed(resolved_conversation_id):
        state = get_session(resolved_conversation_id) or {"conversation": []}
        state["above_threshold"] = inference_out["above_threshold"]
        findings_str = ", ".join(inference_out["above_threshold"]) or "no significant findings"
        state["conversation"].append({
            "role": "system",
            "content": f"[Image analysis ({interaction.id}): {findings_str}] "
                       f"{bundle['llm2_output']['clinical_summary']}",
        })
        save_session(resolved_conversation_id, state)

    return ImageAnalysisResponse(
        interaction_id=interaction.id,
        conversation_id=resolved_conversation_id,
        xray_url=xray_url,
        all_scores=inference_out["all_scores"],
        above_threshold=inference_out["above_threshold"],
        low_confidence_flag=inference_out["low_confidence_flag"],
        gradcam_findings=gradcam_findings_out,
        conditions=[LLMConditionOut(**c) for c in bundle["llm2_output"]["conditions"]],
        clinical_summary=bundle["llm2_output"]["clinical_summary"],
        cross_specialty_notes=bundle["llm2_output"]["cross_specialty_notes"],
        latency_ms=interaction.latency_ms,
    )