"""Analytics page: ask questions in natural language, answered via read-only SQL.

The generated SQL is always shown alongside the result — a specialist reading the query
is the check on whether it actually answered the question asked, which automated
self-review cannot reliably provide.
"""

import pandas as pd
import streamlit as st

from app import api_client

_EXAMPLES = [
    "Berapa banyak kasus per kondisi?",
    "Kondisi apa yang paling sering terdeteksi?",
    "Berapa rata-rata confidence score untuk Pneumothorax?",
    "Berapa banyak temuan dengan GradCAM yang tidak selaras zona anatomis?",
]


def render() -> None:
    """Render the SQL analytics page."""
    st.markdown("## Analytics")
    st.markdown(
        '<span class="ma-caption">Questions are answered by generating read-only SQL against '
        "two curated views. Doctors see only their own records; admins see all.</span>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="ma-divider">', unsafe_allow_html=True)

    with st.expander("Example questions"):
        for example in _EXAMPLES:
            st.markdown(f'<span class="ma-caption">{example}</span>', unsafe_allow_html=True)

    question = st.text_input("Question", placeholder="Berapa banyak kasus Emphysema?", label_visibility="collapsed")
    if not st.button("Run", disabled=not question):
        return

    with st.spinner("Generating and running query…"):
        try:
            result = api_client.agent_query(st.session_state.token, question)
        except api_client.ApiError as exc:
            st.error(str(exc))
            return

    st.markdown(result["explanation"])

    if result["sql_executed"]:
        st.markdown('<span class="ma-mono">generated sql</span>', unsafe_allow_html=True)
        st.code(result["sql_executed"], language="sql")

    if result["rows"]:
        st.dataframe(pd.DataFrame(result["rows"]), use_container_width=True, hide_index=True)
    else:
        st.markdown('<span class="ma-caption">No rows returned.</span>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="ma-caption">{result["row_count"]} rows · {result["latency_ms"]} ms</div>',
        unsafe_allow_html=True,
    )