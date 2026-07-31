"""Chat page: read a study, then keep asking about it.

Upload and text Q&A share one conversation_id, so a follow-up question carries the image
findings with it. Case controls live in the rail rather than the reading column: the column
is for the study and what the system says about it, nothing else.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app import api_client
from app.theme import empty_state
from app.components.chat_bubble import (
    render_user_text, render_user_upload, render_assistant_text,
    render_assistant_record, render_analysis,
)

# Deterministic guidance per rejection reason. The backend already knows exactly why an
# image was refused, so the UI states it plainly instead of surfacing similarity scores
# that mean nothing to a clinician. No LLM is involved in any of these messages.
_UPLOAD_ERROR_GUIDANCE = {
    "not_a_chest_xray": (
        "This file was not recognised as a chest radiograph.",
        "The image was compared against reference chest X-rays and did not match. Common "
        "causes are a different modality (CT slice, ultrasound, MRI), a radiograph of "
        "another body region, a photograph of a screen or printout, or a non-medical image.",
        "Upload a frontal chest radiograph exported directly from PACS as PNG or JPEG.",
    ),
    "outside_training_distribution": (
        "This looks like a chest radiograph, but not one this model can read reliably.",
        "It was recognised as a chest X-ray but sits outside the image distribution the "
        "model was trained on. This usually means a lateral rather than frontal projection, "
        "heavy post-processing or inverted greyscale, a severe crop, or hardware and "
        "acquisition settings unlike the training set.",
        "Try the frontal (PA or AP) projection with the original greyscale and full field "
        "of view, exported without editing.",
    ),
}


def _reset_conversation() -> None:
    """Close the working memory for the open case and clear the thread from view."""
    conversation_id = st.session_state.get("conversation_id")
    if conversation_id:
        try:
            api_client.close_conversation(st.session_state.token, conversation_id)
        except api_client.ApiError:
            pass  # best-effort; the Postgres transcript is preserved regardless
    st.session_state.conversation_id = None
    st.session_state.turns = []


def _render_upload_rejection(code: str | None, fallback_message: str) -> None:
    """Render reason-specific guidance for a rejected upload as a turn in the thread."""
    with st.chat_message("assistant"):
        guidance = _UPLOAD_ERROR_GUIDANCE.get(code)
        if guidance is None:
            st.error(fallback_message)
            return

        headline, explanation, action = guidance
        st.warning(headline)
        st.markdown(explanation)
        st.markdown(f"**What to try:** {action}")
        note = (
            "The case in progress is unchanged — nothing from this file was stored."
            if st.session_state.get("conversation_id")
            else "No case was created and nothing was stored."
        )
        st.markdown(
            f'<span class="ma-caption">{note} Analysis runs only on images that pass '
            "this check.</span>",
            unsafe_allow_html=True,
        )


def _render_case_panel() -> tuple[object, bool]:
    """Render the case controls in the rail; return the uploaded file and whether to analyse."""
    with st.sidebar:
        st.markdown('<div class="ma-rail-label">Case</div>', unsafe_allow_html=True)

        if st.session_state.conversation_id:
            turns = len([t for t in st.session_state.turns if t["kind"] == "analysis"])
            st.markdown(
                f'<div class="ma-caption" style="color:#8C9CB0">Open · '
                f'{turns} stud{"y" if turns == 1 else "ies"} read</div>',
                unsafe_allow_html=True,
            )
            if st.button("End case", width="stretch"):
                _reset_conversation()
                st.rerun()
        else:
            st.markdown(
                '<div class="ma-caption" style="color:#8C9CB0">No case open. Upload a study '
                "to start one.</div>",
                unsafe_allow_html=True,
            )

        st.markdown('<div class="ma-rail-label">Study</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Chest radiograph", type=["png", "jpg", "jpeg"], label_visibility="collapsed",
        )
        analyze = st.button(
            "Analyse study", width="stretch", disabled=uploaded is None, type="primary",
        )
    return uploaded, analyze


def _replay(turns: list[dict]) -> None:
    """Re-render the thread stored in session state."""
    for turn in turns:
        if turn["kind"] == "user_text":
            render_user_text(turn["content"])
        elif turn["kind"] == "user_upload":
            render_user_upload(turn["content"])
        elif turn["kind"] == "assistant_text":
            if turn.get("route") == "records":
                render_assistant_record(
                    turn["answer"], turn.get("rows", []), turn.get("sql_executed"),
                    turn.get("latency_ms"), turn.get("interaction_id"),
                )
            else:
                render_assistant_text(
                    turn["answer"], turn.get("latency_ms"), turn.get("interaction_id"),
                )
        elif turn["kind"] == "analysis":
            render_analysis(turn["result"])
        elif turn["kind"] == "upload_rejected":
            _render_upload_rejection(turn["code"], turn["message"])


def render() -> None:
    """Render the chat page."""
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("turns", [])

    uploaded, analyze_clicked = _render_case_panel()

    if not st.session_state.turns:
        st.markdown(
            empty_state(
                "Read a study",
                "Upload a frontal chest radiograph from the rail. The model screens it for "
                "fourteen findings, localises whatever it reports, and you can question the "
                "result in the same thread.",
            ),
            unsafe_allow_html=True,
        )
    else:
        _replay(st.session_state.turns)

    if analyze_clicked and uploaded is not None:
        render_user_upload(uploaded.name)
        with st.status("Checking the image and preparing the analysis…", expanded=False) as status:
            st.write(
                "The image is verified as a chest radiograph, screened for the 14 covered "
                "findings, and any positive finding is localised and written up."
            )
            try:
                result = api_client.analyze_xray(
                    st.session_state.token, uploaded.getvalue(), uploaded.name,
                    st.session_state.conversation_id,
                )
            except api_client.ApiError as exc:
                status.update(label="Could not analyse this image", state="error")
                # Persist the rejection as a turn. Returning without storing it made the
                # attempt vanish on the next rerun, which read as the thread being reset.
                st.session_state.turns.append({"kind": "user_upload", "content": uploaded.name})
                st.session_state.turns.append(
                    {"kind": "upload_rejected", "code": exc.code, "message": str(exc)}
                )
                st.rerun()
            status.update(label="Analysis complete", state="complete")

        st.session_state.conversation_id = result["conversation_id"]
        st.session_state.turns.append({"kind": "user_upload", "content": uploaded.name})
        st.session_state.turns.append({"kind": "analysis", "result": result})
        st.rerun()

    question = st.chat_input("Ask about this case…")
    if question:
        render_user_text(question)
        with st.spinner("Retrieving and answering…"):
            try:
                result = api_client.ask_question(
                    st.session_state.token, question, st.session_state.conversation_id,
                )
            except api_client.ApiError as exc:
                st.error(str(exc))
                return

        st.session_state.conversation_id = result["conversation_id"]
        st.session_state.turns.append({"kind": "user_text", "content": question})
        st.session_state.turns.append({
            "kind": "assistant_text",
            "interaction_id": result["interaction_id"],
            "answer": result["answer"],
            "latency_ms": result["latency_ms"],
            "route": result.get("route", "clinical"),
            "rows": result.get("rows", []),
            "sql_executed": result.get("sql_executed"),
        })
        st.rerun()