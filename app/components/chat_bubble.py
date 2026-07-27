"""Chat turn renderers: what the reader said, what the system uploaded, and the analysis.

The analysis is laid out as a reading strip. Each finding gets the study and its activation
side by side, because a heatmap only means something against the anatomy underneath it, and
a reader should never have to scroll away from a finding to see what was imaged. The written
explanation sits in the third column so the eye moves image → image → words in one line.
"""

import time

import streamlit as st

from app.theme import condition_badge, image_mount
from app.components.zone_grid import zone_panel
from app.components.confidence_chart import confidence_chart_html, low_confidence_narrative
from app.components.feedback_control import render_feedback


# Prose assembles on first render instead of landing as a block. The backend returns the
# whole answer in one response — there is no token stream to follow — so this is pacing,
# not real streaming: it does not shorten the wait before the first word, it only stops a
# finished paragraph appearing all at once. Evidence (badges, images, zones) is never paced;
# a reader waiting on a study should not wait on an animation to see it.
_REVEAL_WORDS = 3      # words per frame
_REVEAL_DELAY = 0.018  # seconds between frames


def _word_stream(text: str):
    """Yield prose in small groups of words, pacing the reveal."""
    words = text.split(" ")
    for start in range(0, len(words), _REVEAL_WORDS):
        yield " ".join(words[start:start + _REVEAL_WORDS]) + " "
        time.sleep(_REVEAL_DELAY)


def _reveal(text: str, key: str) -> None:
    """Write prose, assembling it the first time and instantly on every rerun after.

    Streamlit replays the whole thread on each interaction, so without the guard every
    click would re-animate the entire conversation.
    """
    revealed = st.session_state.setdefault("revealed", set())
    if key in revealed:
        st.markdown(text)
        return
    revealed.add(key)
    st.write_stream(_word_stream(text))


def render_user_text(text: str) -> None:
    """Render a question the reader asked."""
    with st.chat_message("user"):
        st.markdown(f'<div class="ma-said">{text}</div>', unsafe_allow_html=True)


def render_user_upload(filename: str) -> None:
    """Render the record of a study the reader uploaded."""
    with st.chat_message("user"):
        st.markdown(f'<span class="ma-file">{filename}</span>', unsafe_allow_html=True)


def render_assistant_text(answer: str, cross_specialty_notes: str | None,
                          latency_ms: int | None = None, interaction_id: str | None = None) -> None:
    """Render a written answer, with any cross-specialty note and its feedback control."""
    with st.chat_message("assistant"):
        _reveal(answer, f"{interaction_id}:answer")
        if cross_specialty_notes:
            _render_cross_specialty(cross_specialty_notes)
        _render_latency({"latency_ms": latency_ms})
        if interaction_id:
            render_feedback(interaction_id)


def _render_cross_specialty(note: str) -> None:
    """Render a note that reaches outside chest radiology."""
    st.markdown(
        f'<div style="margin-top:0.9rem;padding-top:0.85rem;border-top:1px solid var(--line)">'
        f'<span class="ma-mono" style="letter-spacing:0.08em;text-transform:uppercase">'
        f"cross-specialty</span>"
        f'<div style="margin-top:0.3rem">{note}</div></div>',
        unsafe_allow_html=True,
    )


def _render_latency(result: dict) -> None:
    """Render the latency readout for a result."""
    if result.get("latency_ms") is not None:
        st.markdown(
            f'<div class="ma-mono" style="margin-top:0.8rem">{result["latency_ms"]} ms</div>',
            unsafe_allow_html=True,
        )


def _alignment_chip(aligned: bool) -> str:
    """Return the chip stating whether activation fell where this finding usually falls."""
    text = "expected distribution" if aligned else "atypical distribution"
    css = "ma-flag-chip" if aligned else "ma-flag-chip is-atypical"
    return f'<span class="{css}"><span class="ma-flag-dot"></span>{text}</span>'


def _render_finding(condition_out: dict, finding: dict | None, xray_url: str,
                    score: float | None, reveal_key: str) -> None:
    """Render one finding as a card: verdict, the three readouts, then the reading.

    The explanation runs full width beneath the images rather than beside them. Text in a
    narrow third column always outruns a square image, which left the image columns padded
    with dead space and stretched every row to the height of its longest sentence.
    """
    with st.container(key=f"finding_{condition_out['name']}"):
        head_left, head_right = st.columns([2, 1.5], vertical_alignment="center")
        with head_left:
            st.markdown(condition_badge(condition_out["name"], score), unsafe_allow_html=True)
        with head_right:
            if finding:
                st.markdown(
                    f'<div style="text-align:right">{_alignment_chip(finding["aligned"])}</div>',
                    unsafe_allow_html=True,
                )

        col_study, col_activation, col_zones = st.columns([1, 1, 0.9], gap="medium")
        with col_study:
            st.markdown(image_mount(xray_url, label="study"), unsafe_allow_html=True)
        with col_activation:
            if finding:
                st.markdown(image_mount(finding["heatmap_url"], label="activation"),
                            unsafe_allow_html=True)
        with col_zones:
            if finding:
                st.markdown(zone_panel(finding["dominant_zones"]), unsafe_allow_html=True)

        # Rendered as markdown, not wrapped in raw HTML: the explanation comes from the LLM
        # and may contain emphasis or lists, which a raw <div> would print as literal text.
        _reveal(condition_out["explanation"], reveal_key)


def render_analysis(result: dict) -> None:
    """Render a full image analysis: findings, activation, zones, and the summary."""
    with st.chat_message("assistant"):
        scores = result["all_scores"]

        if result["low_confidence_flag"]:
            lead, reading = low_confidence_narrative(scores)
            st.markdown("**No finding reached the reporting threshold for this study.**")
            st.markdown(f"{lead} {reading}")

            col_image, col_note = st.columns([1, 2.1], gap="medium")
            with col_image:
                st.markdown(image_mount(result["xray_url"], label="study"), unsafe_allow_html=True)
            with col_note:
                st.markdown(
                    "No heatmaps or zone analysis were produced, because nothing crossed "
                    "threshold to attribute."
                )
                st.markdown(
                    '<div class="ma-caption" style="margin-top:0.6rem">Scope: this model '
                    "reports on 14 chest findings only. Anything outside that set is invisible "
                    "to it, and a below-threshold score is not evidence of absence. Correlation "
                    "with the clinical picture and with prior imaging remains the deciding "
                    "factor.</div>",
                    unsafe_allow_html=True,
                )

            with st.expander("All 14 condition scores", expanded=True):
                st.markdown(confidence_chart_html(scores, result["above_threshold"]),
                            unsafe_allow_html=True)
            _render_latency(result)
            render_feedback(result["interaction_id"])
            return

        badges = "".join(condition_badge(c, scores.get(c)) for c in result["above_threshold"])
        st.markdown(badges, unsafe_allow_html=True)
        with st.expander("All 14 condition scores"):
            st.markdown(confidence_chart_html(scores, result["above_threshold"]),
                        unsafe_allow_html=True)
        st.markdown('<hr class="ma-divider">', unsafe_allow_html=True)

        findings_by_condition = {f["condition"]: f for f in result["gradcam_findings"]}
        for condition_out in result["conditions"]:
            name = condition_out["name"]
            _render_finding(condition_out, findings_by_condition.get(name),
                            result["xray_url"], scores.get(name),
                            f"{result['interaction_id']}:{name}")

        _reveal(result["clinical_summary"], f"{result['interaction_id']}:summary")
        if result["cross_specialty_notes"]:
            _render_cross_specialty(result["cross_specialty_notes"])

        _render_latency(result)
        render_feedback(result["interaction_id"])