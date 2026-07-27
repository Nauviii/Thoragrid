"""Score readouts for the 14 CNN conditions: the full meter list and the near-miss narrative.

`above_threshold` alone hides the case a reader most needs to see — a 0.68 against a 0.70
threshold looks identical to a 0.02. The meter shows every condition; colour is reserved for
the ones that actually passed, so scanning the list still separates signal from noise.
"""

from app.theme import CONDITION_COLORS, TEXT_MUTED

_BELOW_THRESHOLD_COLOR = "#C3CCD6"  # neutral, deliberately not a condition hue

# A top score this far clear of the runner-up means one condition genuinely led; below it,
# the model spread its probability mass rather than favouring any single finding. The cut
# is a presentation heuristic for wording only — it never gates what is displayed.
_DOMINANCE_GAP = 0.15


def low_confidence_narrative(all_scores: dict[str, float], n: int = 3) -> tuple[str, str]:
    """Return (top-scores sentence, distribution reading) for a study where nothing met threshold."""
    ranked = sorted(all_scores.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[:n]
    parts = [f"{condition.replace('_', ' ')} ({score:.2f})" for condition, score in top]
    listed = f"{', '.join(parts[:-1])}, and {parts[-1]}" if len(parts) > 1 else parts[0]
    lead = f"The highest scores were {listed}."

    gap = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0.0
    leader = ranked[0][0].replace("_", " ")
    if gap >= _DOMINANCE_GAP:
        reading = (
            f"{leader} led the rest by a clear margin without reaching threshold, so this "
            f"behaves like a single borderline finding rather than an unremarkable study."
        )
    else:
        reading = (
            "Scores were tightly grouped with no condition standing out, which is the "
            "pattern of a study the model found unremarkable rather than one where a "
            "single finding narrowly missed."
        )
    return lead, reading


def confidence_chart_html(all_scores: dict[str, float], above_threshold: list[str]) -> str:
    """Return HTML for a meter per condition, sorted by score desc, tinted only if above threshold."""
    passed = set(above_threshold)
    ordered = sorted(all_scores.items(), key=lambda kv: kv[1], reverse=True)

    rows = []
    for condition, score in ordered:
        color = CONDITION_COLORS.get(condition, {"text": TEXT_MUTED})["text"] if condition in passed \
            else _BELOW_THRESHOLD_COLOR
        pct = min(max(score, 0.0), 1.0) * 100
        label = condition.replace("_", " ")
        weight = "500" if condition in passed else "400"
        rows.append(
            '<div class="ma-meter-row">'
            f'<span class="ma-meter-name" style="font-weight:{weight}">{label}</span>'
            '<div class="ma-meter-track">'
            f'<div class="ma-meter-fill" style="width:{pct:.1f}%;background:{color}"></div>'
            "</div>"
            f'<span class="ma-meter-value">{score:.2f}</span>'
            "</div>"
        )
    return "".join(rows)