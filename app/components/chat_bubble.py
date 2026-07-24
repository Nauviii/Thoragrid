"""Render one conversation turn: a doctor's message or an assistant response."""

import streamlit as st

from app.theme import condition_badge, TEXT_MUTED
from app.components.zone_grid import zone_grid_svg


def render_user_text(question: str) -> None:
    """Render a doctor's typed question."""
    with st.chat_message("user"):
        st.markdown(question)


def render_user_upload(filename: str) -> None:
    """Render the doctor's X-ray upload as a turn in the conversation."""
    with st.chat_message("user"):
        st.markdown(f'<span class="ma-mono">uploaded &middot; {filename}</span>', unsafe_allow_html=True)


def render_assistant_text(answer: str, cross_specialty_notes: str | None, latency_ms: int | None = None) -> None:
    """Render a text answer from the assistant."""
    with st.chat_message("assistant"):
        st.markdown(answer)
        if cross_specialty_notes:
            st.markdown(
                f'<div style="margin-top:0.7rem;padding-top:0.7rem;border-top:1px solid #E2E6EA">'
                f'<span class="ma-mono">cross-specialty</span><br>{cross_specialty_notes}</div>',
                unsafe_allow_html=True,
            )
        if latency_ms is not None:
            st.markdown(f'<div class="ma-caption">{latency_ms} ms</div>', unsafe_allow_html=True)


def render_analysis(result: dict) -> None:
    """Render a full image analysis result: findings, heatmaps, zone grids, and summary."""
    with st.chat_message("assistant"):
        if result["low_confidence_flag"]:
            st.markdown(result["clinical_summary"])
            st.markdown(
                '<div class="ma-caption">No condition passed its per-class threshold. '
                "This is not the same as a normal study — findings outside the 14 trained "
                "conditions, or below threshold, would also appear this way.</div>",
                unsafe_allow_html=True,
            )
            _render_latency(result)
            return

        scores = result["all_scores"]
        badges = "".join(
            condition_badge(c, scores.get(c)) for c in result["above_threshold"]
        )
        st.markdown(badges, unsafe_allow_html=True)
        st.markdown(f'<hr class="ma-divider">', unsafe_allow_html=True)

        findings_by_condition = {f["condition"]: f for f in result["gradcam_findings"]}

        for condition_out in result["conditions"]:
            name = condition_out["name"]
            finding = findings_by_condition.get(name)

            col_image, col_text = st.columns([1, 2], gap="medium")

            with col_image:
                if finding:
                    st.markdown('<div class="ma-canvas">', unsafe_allow_html=True)
                    st.image(finding["heatmap_url"], use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            with col_text:
                st.markdown(condition_badge(name, scores.get(name)), unsafe_allow_html=True)
                st.markdown(condition_out["explanation"])

                if finding:
                    zones_svg = zone_grid_svg(finding["dominant_zones"])
                    flag_class = "ma-flag-aligned" if finding["aligned"] else "ma-flag-unaligned"
                    flag_text = (
                        "consistent with expected distribution" if finding["aligned"]
                        else "atypical distribution — interpret with caution"
                    )
                    st.markdown(
                        f'<div style="display:flex;gap:0.9rem;align-items:center;margin-top:0.6rem">'
                        f"{zones_svg}"
                        f'<span class="ma-mono {flag_class}">{flag_text}</span></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown('<hr class="ma-divider">', unsafe_allow_html=True)

        st.markdown(result["clinical_summary"])

        if result["cross_specialty_notes"]:
            st.markdown(
                f'<div style="margin-top:0.7rem;padding-top:0.7rem;border-top:1px solid #E2E6EA">'
                f'<span class="ma-mono">cross-specialty</span><br>{result["cross_specialty_notes"]}</div>',
                unsafe_allow_html=True,
            )

        _render_latency(result)


def _render_latency(result: dict) -> None:
    """Render the latency caption for an analysis result."""
    if result.get("latency_ms") is not None:
        st.markdown(f'<div class="ma-caption">{result["latency_ms"]} ms</div>', unsafe_allow_html=True)