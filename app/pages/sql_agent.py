"""Analytics page: ask about the record in plain language, answered by read-only SQL.

The answer leads, then the figures that back it, then — for administrators only — the query
that produced them. The figures stay visible rather than folded away: the sentence above
them is model-written, and the numbers it claims should never be a click out of reach.
"""

import pandas as pd
import streamlit as st

from app import api_client
from app.theme import empty_state
from app.components.result_chart import render_result_chart

_EXAMPLES = [
    "How many cases per condition?",
    "How many findings were flagged as incorrect due to inaccurate localization?",
    "What is the average confidence score for pneumothorax?",
    "How many Grad-CAM findings are misaligned with anatomical zones?",
    "How many findings did I mark as incorrect because the localization was off?",
]

def _render_result(result: dict) -> None:
    """Render the answer, the figures behind it, and the audit trail for administrators."""
    st.markdown(f'<div style="font-size:1.05rem;line-height:1.6">{result["answer"]}</div>',
                unsafe_allow_html=True)

    if result["rows"]:
        st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)
        frame = pd.DataFrame(result["rows"])
        kind = render_result_chart(frame)
        # A headline figure already shows the whole result; repeating it as a one-row table
        # underneath is noise. Every other shape keeps its figures on the page, because the
        # sentence above them is model-written and the numbers it claims should stay checkable.
        if kind != "metric":
            st.dataframe(frame, use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="ma-caption">No records matched.</div>', unsafe_allow_html=True)

    # Query internals are an operator concern, not a clinical one.
    if st.session_state.get("role") == "admin":
        with st.expander("How this was answered"):
            st.markdown(f'<div class="ma-caption">{result["explanation"]}</div>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="ma-caption">{result["row_count"]} matching record(s) · '
                f'{result["latency_ms"]} ms</div>',
                unsafe_allow_html=True,
            )
            if result["sql_executed"]:
                st.code(result["sql_executed"], language="sql")


def render() -> None:
    """Render the analytics page."""
    scope = ("Every reader's records." if st.session_state.get("role") == "admin"
             else "Your own records only.")
    st.markdown(
        '<div class="ma-head"><div>'
        '<div class="ma-head-eyebrow">Record</div>'
        "<h2>Analytics</h2>"
        f'<div class="ma-head-sub">Ask in plain language and get an answer from the reading '
        f"record. {scope}</div></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="ma-divider">', unsafe_allow_html=True)

    question = st.text_input(
        "Question", placeholder="How many cases per condition?", label_visibility="collapsed",
    )
    run = st.button("Ask", disabled=not question, type="primary")

    if not run:
        st.markdown('<div class="ma-rail-label" style="color:var(--muted);margin-top:1.6rem">'
                    "Try asking</div>", unsafe_allow_html=True)
        for example in _EXAMPLES:
            st.markdown(f'<div class="ma-caption" style="padding:0.28rem 0">{example}</div>',
                        unsafe_allow_html=True)
        st.markdown(
            empty_state(
                "Nothing asked yet",
                "Questions are translated into a read-only query against your reading record, "
                "then answered in plain language.",
            ),
            unsafe_allow_html=True,
        )
        return

    with st.spinner("Looking that up…"):
        try:
            result = api_client.agent_query(st.session_state.token, question)
        except api_client.ApiError as exc:
            st.error(str(exc))
            return

    _render_result(result)