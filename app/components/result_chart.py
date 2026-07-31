"""Pick and draw a chart for an analytics result, deterministically, from the data's shape."""

import altair as alt
import pandas as pd

from app.theme import CONDITION_COLORS, LINE, SIGNAL, TEXT_MUTED

_MIN_ROWS_FOR_CHART = 2       # below this a chart adds nothing a line of text doesn't
_VERTICAL_BAR_LIMIT = 5       # above this, upright labels stop fitting
_MAX_BARS = 24                # beyond this the bars are thinner than their own labels
_CHART_HEIGHT = 300
_BAR_THICKNESS = 46           # fixed, so two categories don't become two billboards
_ROW_THICKNESS = 20

_GRID = "#E6EBF0"
_FONT = "Instrument Sans, sans-serif"


def choose_chart(df: pd.DataFrame) -> tuple[str, dict] | None:
    """Return (chart_kind, kwargs) for the frame, or None when a table is the better form."""
    if df.empty:
        return None

    time_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    label_cols = [c for c in df.columns if c not in numeric_cols and c not in time_cols]

    if not numeric_cols:
        return None

    # One row carrying one figure is a headline number. Checked before the two-column rule,
    # because a bare aggregate like avg_confidence arrives as a single unlabelled column and
    # would otherwise fall through to a one-cell table.
    if len(df) == 1 and len(numeric_cols) == 1:
        return "metric", {"label": label_cols[0] if label_cols else None,
                          "value": numeric_cols[0]}

    if len(df.columns) < 2 or len(df) < _MIN_ROWS_FOR_CHART:
        return None

    # A time column means the question was about change over time; a line reads that far
    # better than bars, whatever else came back alongside it.
    if time_cols:
        return "line", {"x": time_cols[0], "y": numeric_cols[0]}

    if label_cols and len(df) <= _MAX_BARS:
        kind = "bar" if len(df) <= _VERTICAL_BAR_LIMIT else "hbar"
        return kind, {"x": label_cols[0], "y": numeric_cols[0]}

    return None


def _humanise(name: str) -> str:
    """Turn a column name into a label a clinician would read."""
    return name.replace("_", " ").strip().capitalize()


def _format_value(value) -> str:
    """Format a headline figure without inventing or dropping precision."""
    if isinstance(value, float):
        return f"{value:,.4g}"
    if isinstance(value, (int,)):
        return f"{value:,}"
    return str(value)


def _condition_scale(labels: pd.Series) -> alt.Scale | None:
    """Return a colour scale keyed to condition hues, or None if the labels aren't conditions."""
    values = [str(v) for v in labels.tolist()]
    if values and all(v in CONDITION_COLORS for v in values):
        return alt.Scale(domain=values, range=[CONDITION_COLORS[v]["text"] for v in values])
    return None


def _themed(chart: alt.Chart) -> alt.Chart:
    """Apply the app's type and hairlines so the chart belongs to the page."""
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            labelFont=_FONT, titleFont=_FONT, labelColor=TEXT_MUTED, titleColor=TEXT_MUTED,
            labelFontSize=11, titleFontSize=11, domainColor=LINE, tickColor=LINE,
            gridColor=_GRID, gridDash=[2, 3],
        )
    )


def _bar_chart(df: pd.DataFrame, label: str, value: str, horizontal: bool) -> alt.Chart:
    """Build a bar chart, coloured per condition where the labels are conditions."""
    scale = _condition_scale(df[label])
    colour = (alt.Color(f"{label}:N", scale=scale, legend=None) if scale is not None
              else alt.value(SIGNAL))
    tooltip = [alt.Tooltip(f"{label}:N", title=_humanise(label)),
               alt.Tooltip(f"{value}:Q", title=_humanise(value))]

    if horizontal:
        return _themed(
            alt.Chart(df)
            .mark_bar(height=_ROW_THICKNESS, cornerRadiusEnd=3)
            .encode(
                x=alt.X(f"{value}:Q", title=_humanise(value)),
                y=alt.Y(f"{label}:N", title=None, sort="-x",
                        axis=alt.Axis(labelLimit=190, labelPadding=8)),
                color=colour, tooltip=tooltip,
            )
            .properties(height=max(_CHART_HEIGHT, _ROW_THICKNESS * 2 * len(df)))
        )

    return _themed(
        alt.Chart(df)
        .mark_bar(size=_BAR_THICKNESS, cornerRadiusEnd=3)
        .encode(
            x=alt.X(f"{label}:N", title=None, sort="-y",
                    axis=alt.Axis(labelAngle=0, labelLimit=150, labelPadding=8)),
            y=alt.Y(f"{value}:Q", title=_humanise(value)),
            color=colour, tooltip=tooltip,
        )
        .properties(height=_CHART_HEIGHT)
    )


def render_result_chart(rows: "pd.DataFrame | list[dict]") -> str | None:
    """Draw the chart that fits this result; return the kind drawn, or None.

    Accepts either a frame or the raw row list an API response carries. The conversion lives
    here rather than at each call site: rows arrive as JSON from the backend, and every caller
    would otherwise repeat the same import and the same one-liner, with the same chance of
    forgetting it.
    """
    import streamlit as st

    if not isinstance(rows, pd.DataFrame):
        if not rows:
            return None
        df = pd.DataFrame(rows)
    else:
        df = rows

    choice = choose_chart(df)
    if choice is None:
        return None

    kind, kwargs = choice

    if kind == "metric":
        row = df.iloc[0]
        label = (str(row[kwargs["label"]]) if kwargs["label"]
                 else _humanise(kwargs["value"]))
        st.metric(label=label, value=_format_value(row[kwargs["value"]]))
        return kind

    if kind == "line":
        st.line_chart(df, x=kwargs["x"], y=kwargs["y"], color=SIGNAL,
                      height=_CHART_HEIGHT, width="stretch")
        return kind

    ordered = df.sort_values(kwargs["y"], ascending=False)
    st.altair_chart(_bar_chart(ordered, kwargs["x"], kwargs["y"], horizontal=(kind == "hbar")),
                    width="stretch")
    return kind