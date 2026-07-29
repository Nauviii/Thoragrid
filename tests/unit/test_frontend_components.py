"""Unit tests for pure presentation helpers (no Streamlit runtime needed)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.components.confidence_chart import confidence_chart_html, low_confidence_narrative
from app.components.result_chart import choose_chart
from app.theme import image_mount

_SCORES = {
    "Atelectasis": 0.66, "Cardiomegaly": 0.36, "Consolidation": 0.56, "Edema": 0.10,
    "Effusion": 0.51, "Emphysema": 0.98, "Fibrosis": 0.05, "Hernia": 0.01,
    "Infiltration": 0.51, "Mass": 0.52, "Nodule": 0.63, "Pleural_Thickening": 0.51,
    "Pneumonia": 0.28, "Pneumothorax": 0.90,
}


def test_confidence_chart_renders_every_condition():
    """All 14 conditions must appear, not just the ones above threshold."""
    html = confidence_chart_html(_SCORES, ["Emphysema", "Pneumothorax"])
    for condition in _SCORES:
        assert condition.replace("_", " ") in html


def test_confidence_chart_sorts_descending():
    """The highest score must appear before the lowest in the rendered order."""
    html = confidence_chart_html(_SCORES, ["Emphysema"])
    assert html.index("Emphysema") < html.index("Hernia")


def test_narrative_names_three_highest_scores():
    """The lead sentence must name the three highest scores and nothing further down."""
    lead, _ = low_confidence_narrative(_SCORES)
    assert "Emphysema (0.98)" in lead
    assert "Pneumothorax (0.90)" in lead
    assert "Atelectasis (0.66)" in lead
    assert "Hernia" not in lead


def test_narrative_reads_a_clear_leader_as_borderline():
    """A top score well clear of the runner-up reads as one borderline finding."""
    _, reading = low_confidence_narrative({"Nodule": 0.66, "Mass": 0.30, "Edema": 0.10})
    assert "clear margin" in reading


def test_narrative_reads_a_flat_spread_as_unremarkable():
    """Tightly grouped scores must not be described as a single borderline finding."""
    _, reading = low_confidence_narrative(
        {"Nodule": 0.50, "Pleural_Thickening": 0.46, "Emphysema": 0.46, "Edema": 0.10}
    )
    assert "tightly grouped" in reading
    assert "clear margin" not in reading


def test_narrative_handles_single_condition():
    """A one-entry score dict must not produce a dangling list separator or divide by zero."""
    lead, reading = low_confidence_narrative({"Mass": 0.4})
    assert lead.count(",") == 0
    assert reading


def test_chart_choice_bar_for_category_counts():
    """A label column plus a numeric column is the bar-chart case."""
    df = pd.DataFrame({"condition": ["A", "B", "C"], "case_count": [5, 3, 1]})
    assert choose_chart(df) == ("bar", {"x": "condition", "y": "case_count"})


def test_chart_choice_line_when_a_time_column_is_present():
    """A datetime column means the question was about change over time."""
    df = pd.DataFrame({"day": pd.to_datetime(["2026-01-01", "2026-01-02"]), "n": [1, 2]})
    assert choose_chart(df) == ("line", {"x": "day", "y": "n"})


def test_chart_choice_metric_for_a_single_figure():
    """One row carrying one number is a headline figure, not a chart."""
    kind, kwargs = choose_chart(pd.DataFrame({"condition": ["Emphysema"], "case_count": [3]}))
    assert kind == "metric"
    assert kwargs == {"label": "condition", "value": "case_count"}


def test_chart_choice_goes_horizontal_once_labels_would_rotate():
    """Condition names are long; past a handful of bars the layout must lay them flat."""
    few = pd.DataFrame({"condition": list("ABC"), "n": [3, 2, 1]})
    many = pd.DataFrame({"condition": [f"Condition {i}" for i in range(12)], "n": range(12)})
    assert choose_chart(few)[0] == "bar"
    assert choose_chart(many)[0] == "hbar"


def test_chart_choice_none_for_shapes_a_table_shows_better():
    """Single columns, non-numeric frames, and empty frames must fall back to the table."""
    assert choose_chart(pd.DataFrame({"condition": ["A", "B"]})) is None
    assert choose_chart(pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})) is None
    assert choose_chart(pd.DataFrame()) is None


def test_image_mount_keeps_wrapper_and_img_together():
    """The mount and the img must be one string; splitting them is the bug this prevents."""
    html = image_mount("https://example.com/x.png")
    assert '<div class="ma-mount">' in html
    assert html.endswith("</div>")
    assert "<img" in html


def test_image_mount_applies_max_width_when_given():
    """max_width constrains the mount so studies and heatmaps render at the same size."""
    assert "max-width:220px" in image_mount("https://example.com/x.png", max_width="220px")
    assert "max-width" not in image_mount("https://example.com/x.png")


def test_image_mount_label_is_optional():
    """A mount without a label must not emit an empty caption element."""
    assert "ma-mount-label" not in image_mount("https://example.com/x.png")
    assert "study" in image_mount("https://example.com/x.png", label="study")


def test_every_reason_code_is_documented_for_the_sql_agent():
    """The UI writes these codes into feedback_comment; the agent must know them to filter on them.

    Without this, renaming or adding a reason would silently produce analytics queries that
    match nothing, with no error anywhere to reveal it.
    """
    from config.feedback_reasons import FEEDBACK_REASONS

    prompt = (Path(__file__).resolve().parent.parent.parent / "core/sql_agent/agent.py").read_text()
    for code in FEEDBACK_REASONS:
        assert code in prompt, f"reason code {code!r} is not documented in the SQL agent prompt"


def test_other_reason_is_part_of_the_reason_set():
    """'other' is treated specially in the UI, so it must still be a real member of the set."""
    from config.feedback_reasons import FEEDBACK_REASONS, OTHER_REASON

    assert OTHER_REASON in FEEDBACK_REASONS


def test_reason_codes_are_query_safe():
    """Codes end up in SQL predicates, so they must stay lowercase identifiers without spaces."""
    from config.feedback_reasons import FEEDBACK_REASONS

    for code in FEEDBACK_REASONS:
        assert code.replace("_", "").isalnum() and code.islower(), code


def test_single_figure_becomes_a_metric_not_a_one_cell_table():
    """A bare aggregate arrives as one unlabelled column; it must not fall through to a table."""
    kind, kwargs = choose_chart(pd.DataFrame({"avg_confidence": [0.9012]}))
    assert kind == "metric"
    assert kwargs == {"label": None, "value": "avg_confidence"}


def test_condition_bars_use_their_own_hues():
    """Bar colour must match the badge and heatmap hue, so colour keeps meaning across the app."""
    from app.components.result_chart import _condition_scale
    from app.theme import CONDITION_COLORS

    scale = _condition_scale(pd.Series(["Emphysema", "Pneumothorax"]))
    assert scale is not None
    assert scale.range == [CONDITION_COLORS["Emphysema"]["text"],
                           CONDITION_COLORS["Pneumothorax"]["text"]]


def test_non_condition_labels_get_no_colour_scale():
    """Hues only mean something for conditions; anything else must not be given a rainbow."""
    from app.components.result_chart import _condition_scale

    assert _condition_scale(pd.Series(["doctor A", "doctor B"])) is None
    assert _condition_scale(pd.Series(["Emphysema", "not a condition"])) is None


def test_bar_chart_spec_builds_with_fixed_thickness():
    """Two categories must not become two billboards; bar width is fixed, not band-filling."""
    from app.components.result_chart import _bar_chart

    frame = pd.DataFrame({"condition": ["Emphysema", "Pneumothorax"], "case_count": [22, 22]})
    spec = _bar_chart(frame, "condition", "case_count", horizontal=False).to_dict()
    assert spec["mark"]["size"] == 46