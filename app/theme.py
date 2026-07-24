"""Design tokens and injected CSS for the MedAssist frontend.

Visual direction: light, airy clinical tooling — cool neutrals evoking a film lightbox
rather than warm cream or dark dashboard defaults. The only dark surface is the X-ray
canvas itself, mirroring how real PACS viewers keep application chrome light while the
reading area stays dark for contrast sensitivity.

Condition colors are derived from settings.gradcam_condition_colors so the chat UI and
the GradCAM overlays speak the same color language: the same hue identifies a condition
whether it appears as a heatmap tint or a badge.
"""

BACKGROUND = "#F7F8FA"
SURFACE = "#FFFFFF"
TEXT_PRIMARY = "#1C2530"
TEXT_MUTED = "#6B7684"
ACCENT = "#2B4C5C"
ACCENT_SOFT = "#EAF0F2"
BORDER = "#E2E6EA"
CANVAS_DARK = "#161B22"

# Derived from settings.gradcam_condition_colors: text is the hue darkened for legibility,
# background is the same hue at ~13% alpha to stay light.
CONDITION_COLORS = {
    "Atelectasis":        {"text": "#8C0000", "bg": "#FF000022"},
    "Cardiomegaly":       {"text": "#00008C", "bg": "#0000FF22"},
    "Consolidation":      {"text": "#8C0B50", "bg": "#FF149322"},
    "Edema":              {"text": "#104F8C", "bg": "#1E90FF22"},
    "Effusion":           {"text": "#008C00", "bg": "#00FF0022"},
    "Emphysema":          {"text": "#8C7600", "bg": "#FFD70022"},
    "Fibrosis":           {"text": "#4C250A", "bg": "#8B451322"},
    "Hernia":             {"text": "#464600", "bg": "#80800022"},
    "Infiltration":       {"text": "#8C8C00", "bg": "#FFFF0022"},
    "Mass":               {"text": "#8C008C", "bg": "#FF00FF22"},
    "Nodule":             {"text": "#008C8C", "bg": "#00FFFF22"},
    "Pleural_Thickening": {"text": "#004646", "bg": "#00808022"},
    "Pneumonia":          {"text": "#8C4600", "bg": "#FF800022"},
    "Pneumothorax":       {"text": "#46008C", "bg": "#8000FF22"},
}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

:root {{
    --ma-bg: {BACKGROUND};
    --ma-surface: {SURFACE};
    --ma-text: {TEXT_PRIMARY};
    --ma-muted: {TEXT_MUTED};
    --ma-accent: {ACCENT};
    --ma-accent-soft: {ACCENT_SOFT};
    --ma-border: {BORDER};
    --ma-canvas: {CANVAS_DARK};
}}

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', -apple-system, sans-serif;
    color: var(--ma-text);
}}

.stApp {{ background: var(--ma-bg); }}

/* Tighten Streamlit's default vertical padding — the stock spacing reads as "demo app" */
.block-container {{ padding-top: 2.2rem; padding-bottom: 6rem; max-width: 62rem; }}

h1, h2, h3 {{ font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; letter-spacing: -0.01em; }}
h1 {{ font-size: 1.75rem; }}

/* App header: hairline rule, not a heavy bar */
.ma-header {{
    display: flex; align-items: baseline; justify-content: space-between;
    padding-bottom: 0.9rem; margin-bottom: 1.6rem;
    border-bottom: 1px solid var(--ma-border);
}}
.ma-header-title {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.3rem; font-weight: 600; color: var(--ma-text);
}}
.ma-header-meta {{ font-size: 0.8rem; color: var(--ma-muted); }}

/* Chat turns */
[data-testid="stChatMessage"] {{
    background: var(--ma-surface);
    border: 1px solid var(--ma-border);
    border-radius: 10px;
    box-shadow: 0 1px 2px rgba(28, 37, 48, 0.04);
    padding: 1rem 1.15rem;
}}

/* Condition badge — tinted pill, never a solid block */
.ma-badge {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.2rem 0.6rem; margin: 0.15rem 0.3rem 0.15rem 0;
    border-radius: 999px; font-size: 0.78rem; font-weight: 500;
}}
.ma-badge-score {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; opacity: 0.85; }}

/* Numeric/technical values read as measured data, distinct from prose */
.ma-mono {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: var(--ma-muted); }}

/* X-ray canvas: the one intentionally dark surface */
.ma-canvas {{
    background: var(--ma-canvas);
    border-radius: 10px; padding: 0.75rem;
    border: 1px solid #232A33;
}}

.ma-caption {{ font-size: 0.75rem; color: var(--ma-muted); margin-top: 0.35rem; }}
.ma-divider {{ height: 1px; background: var(--ma-border); border: 0; margin: 1.4rem 0; }}

/* Alignment flag — muted by default, amber only when clinically noteworthy */
.ma-flag-aligned {{ color: var(--ma-muted); }}
.ma-flag-unaligned {{ color: #9A6B00; }}

/* Buttons: quiet, hairline-bordered, no heavy fills */
.stButton > button {{
    background: var(--ma-surface); color: var(--ma-text);
    border: 1px solid var(--ma-border); border-radius: 8px;
    font-size: 0.85rem; font-weight: 500; padding: 0.4rem 0.9rem;
    transition: border-color 120ms ease, background 120ms ease;
}}
.stButton > button:hover {{ border-color: var(--ma-accent); background: var(--ma-accent-soft); color: var(--ma-accent); }}

[data-testid="stChatInput"] {{ border: 1px solid var(--ma-border); border-radius: 10px; background: var(--ma-surface); }}

[data-testid="stSidebar"] {{ background: var(--ma-surface); border-right: 1px solid var(--ma-border); }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}

#MainMenu, footer, header {{ visibility: hidden; }}

.stDataFrame {{ border: 1px solid var(--ma-border); border-radius: 8px; }}
</style>
"""


def inject_css() -> None:
    """Inject the app's custom CSS; call once per page render."""
    import streamlit as st
    st.markdown(_CSS, unsafe_allow_html=True)


def condition_badge(condition: str, score: float | None = None) -> str:
    """Return HTML for a condition badge tinted with that condition's GradCAM hue."""
    colors = CONDITION_COLORS.get(condition, {"text": TEXT_MUTED, "bg": ACCENT_SOFT})
    label = condition.replace("_", " ")
    score_html = f'<span class="ma-badge-score">{score:.2f}</span>' if score is not None else ""
    return (
        f'<span class="ma-badge" style="background:{colors["bg"]};color:{colors["text"]}">'
        f"{label}{score_html}</span>"
    )