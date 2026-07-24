"""Unified chat page: upload an X-ray and follow up in the same thread.

Upload and text Q&A share one conversation_id, so a follow-up question carries the image
findings as context — the backend handles that continuity; this page just keeps the id in
session state and renders the turns.
"""

import streamlit as st

from app import api_client
from app.components.chat_bubble import (
    render_user_text, render_user_upload, render_assistant_text, render_analysis,
)


def _reset_conversation() -> None:
    """Start a fresh conversation, closing the previous one's working memory."""
    conversation_id = st.session_state.get("conversation_id")
    if conversation_id:
        try:
            api_client.close_conversation(st.session_state.token, conversation_id)
        except api_client.ApiError:
            pass  # closing is best-effort; the transcript is preserved regardless
    st.session_state.conversation_id = None
    st.session_state.turns = []


def render() -> None:
    """Render the chat page."""
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("turns", [])

    with st.sidebar:
        st.markdown("### Case")
        if st.session_state.conversation_id:
            st.markdown(
                f'<span class="ma-mono">{st.session_state.conversation_id[:8]}…</span>',
                unsafe_allow_html=True,
            )
            if st.button("End case", use_container_width=True):
                _reset_conversation()
                st.rerun()
        else:
            st.markdown('<span class="ma-caption">No active case</span>', unsafe_allow_html=True)

        st.markdown('<hr class="ma-divider">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Chest X-ray", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        analyze_clicked = st.button("Analyze X-ray", use_container_width=True, disabled=uploaded is None)

    for turn in st.session_state.turns:
        if turn["kind"] == "user_text":
            render_user_text(turn["content"])
        elif turn["kind"] == "user_upload":
            render_user_upload(turn["content"])
        elif turn["kind"] == "assistant_text":
            render_assistant_text(turn["answer"], turn["cross_specialty_notes"], turn.get("latency_ms"))
        elif turn["kind"] == "analysis":
            render_analysis(turn["result"])

    if analyze_clicked and uploaded is not None:
        render_user_upload(uploaded.name)
        with st.spinner("Running CNN, GradCAM, retrieval, and clinical synthesis…"):
            try:
                result = api_client.analyze_xray(
                    st.session_state.token, uploaded.getvalue(), uploaded.name,
                    st.session_state.conversation_id,
                )
            except api_client.ApiError as exc:
                st.error(str(exc))
                return

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
            "answer": result["answer"],
            "cross_specialty_notes": result["cross_specialty_notes"],
            "latency_ms": result["latency_ms"],
        })
        st.rerun()