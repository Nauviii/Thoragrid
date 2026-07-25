"""Render all 14 CNN condition scores as a sorted horizontal bar chart, for the near-miss cases
`above_threshold` alone can't show (e.g. a 0.68 score against a 0.70 threshold).
"""

from app.theme import CONDITION_COLORS, TEXT_MUTED

_BELOW_THRESHOLD_COLOR = "#C6CDD4"  # neutral gray, matches theme.BORDER family


def confidence_chart_html(all_scores: dict[str, float], above_threshold: list[str]) -> str:
    """Return HTML for a bar per condition, sorted by score desc, tinted only if above threshold."""
    passed = set(above_threshold)
    ordered = sorted(all_scores.items(), key=lambda kv: kv[1], reverse=True)

    rows = []
    for condition, score in ordered:
        color = CONDITION_COLORS.get(condition, {"text": TEXT_MUTED})["text"] if condition in passed \
            else _BELOW_THRESHOLD_COLOR
        pct = min(max(score, 0.0), 1.0) * 100
        label = condition.replace("_", " ")
        rows.append(
            '<div style="display:flex;align-items:center;gap:0.6rem;margin:0.22rem 0">'
            f'<span class="ma-mono" style="width:9.5rem;flex-shrink:0">{label}</span>'
            '<div style="flex:1;background:#EEF1F3;border-radius:4px;height:0.55rem">'
            f'<div style="width:{pct:.1f}%;background:{color};height:100%;border-radius:4px"></div>'
            "</div>"
            f'<span class="ma-mono" style="width:2.6rem;text-align:right">{score:.2f}</span>'
            "</div>"
        )
    return "".join(rows)