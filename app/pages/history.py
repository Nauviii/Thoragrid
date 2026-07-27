"""History page: browse past readings and open the full thread behind any of them.

Rows are lazy. The thread behind a row is fetched only when that row is opened, because
Streamlit executes the body of a collapsed expander just like an open one — the previous
version therefore fetched every conversation, and re-signed every stored image, on every
single page load. A reader scanning twenty rows paid for twenty threads to see none.

The first page renders at once and the remainder fills in behind it, so a long record is
readable immediately instead of after the whole list resolves.
"""

import time

import streamlit as st

from app import api_client
from app.theme import condition_badge, image_mount, empty_state
from app.components.zone_grid import zone_panel
from app.components.feedback_control import render_feedback

_FIRST_PAGE = 10     # rows shown before the reveal starts
_BATCH = 4           # rows added per frame afterwards
_BATCH_DELAY = 0.09  # seconds between frames

_VERDICT_MARK = {True: "agreed", False: "flagged"}


def _render_image_turn(turn: dict) -> None:
    """Render an image turn: study, per-condition activation, tolerating expired signed URLs."""
    conditions = turn.get("above_threshold") or []
    badges = "".join(condition_badge(c) for c in conditions) or (
        '<span class="ma-caption">No finding above threshold</span>'
    )
    st.markdown(badges, unsafe_allow_html=True)

    col_study, col_rest = st.columns([1, 2.1], gap="medium")
    with col_study:
        if turn.get("xray_url"):
            st.markdown(image_mount(turn["xray_url"], label="study"), unsafe_allow_html=True)
        else:
            st.markdown('<span class="ma-caption">Study image is no longer available.</span>',
                        unsafe_allow_html=True)
    with col_rest:
        if turn.get("clinical_summary"):
            st.markdown(turn["clinical_summary"])

    for finding in turn.get("gradcam_findings") or []:
        col_image, col_zones, col_name = st.columns([1, 0.9, 1.2], gap="medium")
        with col_image:
            if finding["heatmap_url"]:
                st.markdown(image_mount(finding["heatmap_url"], label="activation"),
                            unsafe_allow_html=True)
            else:
                st.markdown('<span class="ma-caption">Heatmap is no longer available.</span>',
                            unsafe_allow_html=True)
        with col_zones:
            st.markdown(zone_panel(finding["dominant_zones"]), unsafe_allow_html=True)
        with col_name:
            st.markdown(condition_badge(finding["condition"]), unsafe_allow_html=True)


def _render_transcript(conversation_id: str) -> None:
    """Render a full conversation thread from the permanent Postgres record."""
    try:
        transcript = api_client.get_conversation(st.session_state.token, conversation_id)
    except api_client.ApiError as exc:
        st.error(str(exc))
        return

    for turn in transcript["turns"]:
        if turn["interaction_type"] == "text":
            st.markdown(f'<div class="ma-said" style="font-weight:600">{turn["query"] or "—"}</div>',
                        unsafe_allow_html=True)
            if turn["answer"]:
                st.markdown(turn["answer"])
        else:
            _render_image_turn(turn)
        st.markdown('<hr class="ma-divider">', unsafe_allow_html=True)


def _row_label(item: dict) -> str:
    """Build the row label: what was found first, then when, then how it was handled."""
    timestamp = item["timestamp"].replace("T", " ")[:16]
    if item["interaction_type"] == "image":
        lead = ", ".join(item.get("above_threshold") or []) or "No finding above threshold"
    else:
        lead = (item["raw_query"] or "Question")[:64]

    verdict = _VERDICT_MARK.get(item.get("feedback"))
    return f"{lead}   ·   {timestamp}" + (f"   ·   {verdict}" if verdict else "")


def _render_row(item: dict) -> None:
    """Render one collapsed row; fetch and show its thread only once it is opened."""
    open_key = f"hist_open_{item['id']}"
    is_open = st.session_state.get(open_key, False)

    with st.container(key=f"histrow_{item['id']}"):
        icon = ":material/expand_less:" if is_open else ":material/expand_more:"
        if st.button(_row_label(item), key=f"histbtn_{item['id']}", icon=icon,
                     use_container_width=True):
            st.session_state[open_key] = not is_open
            st.rerun()

        if is_open:
            st.markdown(
                f'<div class="ma-mono" style="margin:0.5rem 0 0.2rem">'
                f'{item["interaction_type"]} · {item["latency_ms"] or "—"} ms</div>',
                unsafe_allow_html=True,
            )
            _render_transcript(item["conversation_id"])
            render_feedback(item["id"], recorded=item.get("feedback"))


def render() -> None:
    """Render the history page."""
    st.markdown(
        '<div class="ma-head"><div>'
        '<div class="ma-head-eyebrow">Record</div>'
        "<h2>History</h2>"
        '<div class="ma-head-sub">Every reading is kept after its case closes. Closing a case '
        "clears working memory only, never the record. Open a row to load its thread.</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="ma-divider">', unsafe_allow_html=True)

    try:
        history = api_client.get_history(st.session_state.token)
    except api_client.ApiError as exc:
        st.error(str(exc))
        return

    items = history["items"]
    if not items:
        st.markdown(
            empty_state(
                "Nothing read yet",
                "Studies you analyse and questions you ask will collect here, with the full "
                "thread behind each one.",
            ),
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="ma-mono" style="margin-bottom:0.7rem">{history["total"]} '
        f"interaction{'s' if history['total'] != 1 else ''} on record</div>",
        unsafe_allow_html=True,
    )

    for item in items[:_FIRST_PAGE]:
        _render_row(item)

    rest = items[_FIRST_PAGE:]
    if not rest:
        return

    # Pace the remainder on the first visit only. Re-animating on every expand would make
    # opening a row near the bottom of a long record feel broken.
    if st.session_state.get("history_revealed"):
        for item in rest:
            _render_row(item)
        return

    for start in range(0, len(rest), _BATCH):
        for item in rest[start:start + _BATCH]:
            _render_row(item)
        time.sleep(_BATCH_DELAY)
    st.session_state.history_revealed = True