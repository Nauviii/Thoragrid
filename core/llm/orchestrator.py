"""Orchestrate LLM Call 1, RAG retrieval, caching, and LLM Call 2 for the image analysis path."""

from pinecone import Index

from config.settings import settings
from core.rag.retriever import retrieve_for_image_path, retrieve_for_text_path
from core.llm.client import call_groq
from core.llm.prompts import (
    LLM1_SYSTEM, LLM1_SCHEMA, build_llm1_user_prompt, parse_llm1_output,
    LLM2_SYSTEM, LLM2_SCHEMA, build_llm2_user_prompt, parse_llm2_output,
    TEXT_QA_SYSTEM, TEXT_QA_SCHEMA, build_text_qa_user_prompt, parse_text_qa_output,
)
from core.llm.guardrails import (
    validate_llm2_output, validate_text_qa_output,
    check_prompt_injection, sanitize_user_input,
)
from core.llm.cache import get_cached, set_cached, get_cached_text, set_cached_text

NO_FINDING_BUNDLE = {
    "llm1_output": None,
    "rag_chunks": [],
    "llm2_output": {
        "conditions": [],
        "clinical_summary": "No significant findings detected above threshold.",
        "cross_specialty_notes": None,
    },
}


def run_image_llm_pipeline(
    all_scores: dict[str, float],
    above_threshold: list[str],
    low_confidence_flag: bool,
    gradcam_results: dict,
    semantic_context: str,
    index: Index,
    namespace: str,
) -> dict:
    """Run LLM Call 1, retrieval, cache check, and LLM Call 2; returns full bundle for audit logging.

    Precondition: above_threshold must be the same list passed to run_gradcam(), so every
    condition in above_threshold has a matching entry in gradcam_results.
    """
    if low_confidence_flag:
        return NO_FINDING_BUNDLE

    cached = get_cached("pipeline", above_threshold, gradcam_results)
    if cached is not None:
        return cached

    llm1_user = build_llm1_user_prompt(above_threshold, all_scores, semantic_context)
    llm1_raw  = call_groq(LLM1_SYSTEM, llm1_user, schema=LLM1_SCHEMA, schema_name="rag_queries")
    llm1_out  = parse_llm1_output(llm1_raw)

    rag_chunks = retrieve_for_image_path(llm1_out["rag_queries"], index, namespace)

    llm2_user = build_llm2_user_prompt(above_threshold, all_scores, semantic_context, rag_chunks)
    llm2_raw  = call_groq(LLM2_SYSTEM, llm2_user, schema=LLM2_SCHEMA,
                          schema_name="clinical_explanation",
                          max_tokens=settings.llm2_max_tokens)
    llm2_out  = parse_llm2_output(llm2_raw)

    if not validate_llm2_output(llm2_out):
        raise ValueError("LLM Call 2 output failed clinical safety validation")

    bundle = {"llm1_output": llm1_out, "rag_chunks": rag_chunks, "llm2_output": llm2_out}
    set_cached("pipeline", above_threshold, gradcam_results, bundle)
    return bundle


REJECTED_QUERY_RESPONSE = {
    "answer": "Query rejected: input did not pass safety validation.",
}


def run_text_llm_pipeline(
    query: str,
    index: Index,
    namespace: str,
    top_k: int = 4,
    prior_context: dict | None = None,
) -> dict:
    """Run guardrail check, retrieval, cache check, and LLM call; returns bundle for audit logging.

    When prior_context is given (a follow-up in an ongoing conversation), the pattern-based
    cache is skipped entirely — cached answers are cross-patient by design and would be wrong
    for a conversation-specific follow-up. The retrieval query is also enriched with prior
    finding condition names so RAG stays anchored to the same case.
    """
    query = sanitize_user_input(query)

    if check_prompt_injection(query):
        return {"query_used": query, "rag_chunks": [], "answer_output": REJECTED_QUERY_RESPONSE}

    is_followup = prior_context is not None

    if not is_followup:
        cached = get_cached_text(query)
        if cached is not None:
            return cached

    retrieval_query = query
    if is_followup:
        above = prior_context.get("above_threshold") or []
        if above:
            retrieval_query = f"{query} ({', '.join(above)})"

    rag_chunks = retrieve_for_text_path(retrieval_query, index, namespace, top_k)

    user_prompt = build_text_qa_user_prompt(query, rag_chunks, prior_context)
    raw = call_groq(TEXT_QA_SYSTEM, user_prompt, schema=TEXT_QA_SCHEMA, schema_name="text_qa_answer")
    parsed = parse_text_qa_output(raw)

    if not validate_text_qa_output(parsed):
        raise ValueError("Text Q&A output failed clinical safety validation")

    bundle = {"query_used": query, "rag_chunks": rag_chunks, "answer_output": parsed}
    if not is_followup:
        set_cached_text(query, bundle)
    return bundle