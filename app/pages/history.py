"""History page: browse past interactions and open full conversation transcripts."""

import streamlit as st

from app import api_client
from app.theme import condition_badge, image_canvas
from app.components.zone_grid import zone_grid_svg


def _render_image_turn(turn: dict) -> None:
    """Render an image-type turn: X-ray + per-condition heatmaps, tolerating expired signed URLs."""
    conditions = turn.get("above_threshold") or []
    badges = "".join(condition_badge(c) for c in conditions) or (
        '<span class="ma-caption">No findings above threshold</span>'
    )
    st.markdown(badges, unsafe_allow_html=True)

    if turn.get("xray_url"):
        st.markdown(image_canvas(turn["xray_url"], max_width="220px"), unsafe_allow_html=True)
    else:
        st.markdown('<span class="ma-caption">X-ray image no longer available</span>', unsafe_allow_html=True)

    for finding in turn.get("gradcam_findings") or []:
        col_image, col_meta = st.columns([1, 2], gap="small")
        with col_image:
            if finding["heatmap_url"]:
                st.markdown(image_canvas(finding["heatmap_url"]), unsafe_allow_html=True)
            else:
                st.markdown('<span class="ma-caption">Heatmap no longer available</span>', unsafe_allow_html=True)
        with col_meta:
            st.markdown(condition_badge(finding["condition"]), unsafe_allow_html=True)
            st.markdown(zone_grid_svg(finding["dominant_zones"]), unsafe_allow_html=True)

    if turn.get("clinical_summary"):
        st.markdown(turn["clinical_summary"])


def _render_transcript(conversation_id: str) -> None:
    """Render a full conversation transcript, read from the permanent Postgres record."""
    try:
        transcript = api_client.get_conversation(st.session_state.token, conversation_id)
    except api_client.ApiError as exc:
        st.error(str(exc))
        return

    for turn in transcript["turns"]:
        if turn["interaction_type"] == "text":
            st.markdown(f"**{turn['query'] or '—'}**")
            if turn["answer"]:
                st.markdown(turn["answer"])
        else:
            _render_image_turn(turn)
        st.markdown('<hr class="ma-divider">', unsafe_allow_html=True)


def render() -> None:
    """Render the history page."""
    st.markdown("## History")
    st.markdown(
        '<span class="ma-caption">Past interactions remain readable after a case is closed — '
        "closing only clears working memory, never the record.</span>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="ma-divider">', unsafe_allow_html=True)

    st.session_state.setdefault("feedback_done", set())

    try:
        history = api_client.get_history(st.session_state.token)
    except api_client.ApiError as exc:
        st.error(str(exc))
        return

    if not history["items"]:
        st.markdown('<span class="ma-caption">No interactions yet.</span>', unsafe_allow_html=True)
        return

    for item in history["items"]:
        timestamp = item["timestamp"].replace("T", " ")[:19]
        label = item["raw_query"] or ", ".join(item.get("above_threshold") or []) or "No findings"
        header = f'{item["interaction_type"]} · {timestamp} · {label[:60]}'

        with st.expander(header):
            st.markdown(
                f'<span class="ma-mono">conversation {item["conversation_id"][:8]}… · '
                f'{item["latency_ms"] or "—"} ms</span>',
                unsafe_allow_html=True,
            )
            st.markdown('<hr class="ma-divider">', unsafe_allow_html=True)
            _render_transcript(item["conversation_id"])

            already_done = item["id"] in st.session_state.feedback_done
            col_agree, col_disagree, col_status = st.columns([1, 1, 3])
            with col_agree:
                if st.button("Agree", key=f"agree_{item['id']}", use_container_width=True, disabled=already_done):
                    _submit_feedback(item["id"], True)
            with col_disagree:
                if st.button("Disagree", key=f"disagree_{item['id']}", use_container_width=True, disabled=already_done):
                    _submit_feedback(item["id"], False)
            with col_status:
                if already_done:
                    st.markdown('<span class="ma-caption">Feedback recorded for this interaction.</span>',
                                unsafe_allow_html=True)


def _submit_feedback(interaction_id: str, is_correct: bool) -> None:
    """Submit agree/disagree feedback for one interaction; a 409 means it was already recorded."""
    try:
        api_client.submit_feedback(st.session_state.token, interaction_id, is_correct)
        st.session_state.feedback_done.add(interaction_id)
        st.rerun()
    except api_client.ApiError as exc:
        if str(exc).startswith("409"):
            st.session_state.feedback_done.add(interaction_id)
            st.info("Feedback was already recorded for this interaction.")
            st.rerun()
        else:
            st.warning(str(exc))