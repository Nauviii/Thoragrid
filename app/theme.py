"""Design system for MedAssist: tokens, type, CSS, and the marks used across the app.

Direction — "reading room". Radiologists read in a dimmed room against a bright film, so the
app is built the same way: a deep ink shell holds navigation and identity, and the reading
surface it surrounds is luminous. That inversion is the point. A uniformly bright interface
fights the way this audience actually works, and a uniformly dark one loses the film.

Identity comes from the product's own instrument rather than decoration. The mark is the
seven-zone thorax schematic with a single zone lit — the same readout `zone_grid.py` draws
from real GradCAM measurements. The motif recurs at three scales: the logo, the per-finding
zone chip, and a watermark behind empty states.

Condition hues stay derived from settings.gradcam_condition_colors, unchanged: a badge and
its heatmap overlay must speak the same colour or the pairing stops being readable.
"""

from urllib.parse import quote

# --- Palette -----------------------------------------------------------------------------
INK          = "#0E1621"   # the darkened room: shell, image mounts
INK_RAISED   = "#18222F"   # raised surfaces inside the shell
INK_LINE     = "#2A3644"   # hairlines on ink
FILM         = "#EFF3F6"   # the diffuser glow of the reading surface
SURFACE      = "#FFFFFF"   # cards on film
LINE         = "#DCE3EA"   # hairlines on film
TEXT         = "#101A24"
TEXT_MUTED   = "#5C6A7C"  # 4.94:1 on film, clears AA for body text
SIGNAL       = "#0E7C86"   # deep clinical teal: primary action, active state
SIGNAL_SOFT  = "#E1F0F1"
HALO         = "#6FD3DA"   # light teal, only ever on ink
CAUTION      = "#A96400"   # reserved for atypical distribution, nothing else

# Derived from settings.gradcam_condition_colors: text is the hue darkened for legibility on
# white, background the same hue at low alpha. Kept in lockstep with the GradCAM overlays.
CONDITION_COLORS = {
    "Atelectasis":        {"text": "#8C0000", "bg": "#FF00001A"},
    "Cardiomegaly":       {"text": "#00008C", "bg": "#0000FF1A"},
    "Consolidation":      {"text": "#8C0B50", "bg": "#FF14931A"},
    "Edema":              {"text": "#104F8C", "bg": "#1E90FF1A"},
    "Effusion":           {"text": "#046604", "bg": "#00FF001A"},
    "Emphysema":          {"text": "#7A6700", "bg": "#FFD7001A"},
    "Fibrosis":           {"text": "#4C250A", "bg": "#8B45131A"},
    "Hernia":             {"text": "#464600", "bg": "#8080001A"},
    "Infiltration":       {"text": "#7A7A00", "bg": "#FFFF001A"},
    "Mass":               {"text": "#8C008C", "bg": "#FF00FF1A"},
    "Nodule":             {"text": "#006969", "bg": "#00FFFF1A"},
    "Pleural_Thickening": {"text": "#004646", "bg": "#0080801A"},
    "Pneumonia":          {"text": "#8C4600", "bg": "#FF80001A"},
    "Pneumothorax":       {"text": "#46008C", "bg": "#8000FF1A"},
}


# --- Marks -------------------------------------------------------------------------------
def _svg_data_uri(svg: str) -> str:
    """Encode an SVG for use in a CSS url(); quoting beats base64 for readable diffs."""
    return f"data:image/svg+xml;utf8,{quote(svg)}"


def brand_mark(size: int = 32) -> str:
    """The MedAssist mark: a chest field divided into zones, one of them lit.

    Two divisions rather than the full seven — at 32px the complete schematic turns to mud,
    and the meaning survives the abstraction.
    """
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 32 32" fill="none" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="MedAssist">'
        f'<rect width="32" height="32" rx="9" fill="{INK}"/>'
        f'<g stroke="{HALO}" stroke-opacity="0.4" stroke-width="1.25" '
        f'stroke-linecap="round" fill="none">'
        f'<rect x="8" y="7" width="16" height="18" rx="2.5"/>'
        f'<path d="M16 7.6v16.8"/><path d="M8.6 13h14.8"/><path d="M8.6 19h14.8"/>'
        f"</g>"
        f'<rect x="16.85" y="13.85" width="6.3" height="4.3" rx="1.3" fill="{HALO}"/>'
        f"</svg>"
    )


def brand_lockup() -> str:
    """The mark paired with the wordmark, for the sidebar head and the sign-in screen."""
    return (
        f'<div class="ma-brand">{brand_mark(30)}'
        f'<span class="ma-brand-word">MedAssist</span></div>'
    )


def zone_watermark(size: int = 220) -> str:
    """An oversized, very faint zone schematic used behind empty states."""
    return (
        f'<svg width="{size}" height="{size * 1.15:.0f}" viewBox="0 0 100 115" fill="none" '
        f'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        f'<g stroke="{TEXT_MUTED}" stroke-opacity="0.16" stroke-width="1" fill="none">'
        f'<rect x="4" y="4" width="92" height="107" rx="6"/>'
        f'<path d="M50 5v105"/><path d="M5 42h90"/><path d="M5 74h90"/>'
        f'<rect x="33" y="35" width="34" height="46" rx="4"/>'
        f"</g></svg>"
    )


_AVATAR_ASSISTANT = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">'
    f'<rect width="32" height="32" rx="10" fill="{INK}"/>'
    f'<g stroke="{HALO}" stroke-opacity="0.45" stroke-width="1.3" fill="none" '
    'stroke-linecap="round">'
    '<rect x="9" y="8" width="14" height="16" rx="2.4"/>'
    '<path d="M16 8.6v14.8"/><path d="M9.6 13.6h12.8"/><path d="M9.6 18.6h12.8"/></g>'
    f'<rect x="16.8" y="14.4" width="5.4" height="3.6" rx="1.1" fill="{HALO}"/></svg>'
)

_AVATAR_USER = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">'
    f'<rect width="32" height="32" rx="10" fill="{SIGNAL_SOFT}"/>'
    f'<circle cx="16" cy="13" r="4.1" fill="none" stroke="{SIGNAL}" stroke-width="1.5"/>'
    f'<path d="M8.6 25c1.4-4 4.1-6 7.4-6s6 2 7.4 6" fill="none" stroke="{SIGNAL}" '
    'stroke-width="1.5" stroke-linecap="round"/></svg>'
)


# --- Stylesheet --------------------------------------------------------------------------
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:ital,wght@0,400;0,500;0,600;1,400&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
  --ink: {INK}; --ink-raised: {INK_RAISED}; --ink-line: {INK_LINE};
  --film: {FILM}; --surface: {SURFACE}; --line: {LINE};
  --text: {TEXT}; --muted: {TEXT_MUTED};
  --signal: {SIGNAL}; --signal-soft: {SIGNAL_SOFT}; --halo: {HALO}; --caution: {CAUTION};
  --sans: 'Instrument Sans', ui-sans-serif, system-ui, sans-serif;
  --serif: 'Instrument Serif', Georgia, serif;
  --mono: 'JetBrains Mono', ui-monospace, monospace;
  --r-sm: 8px; --r-md: 12px; --r-lg: 16px;
}}

html, body, [class*="css"], .stApp {{
  font-family: var(--sans);
  color: var(--text);
  font-feature-settings: 'ss01', 'cv01';
}}
.stApp {{ background: var(--film); }}

.stMainBlockContainer {{ padding-top: 2.6rem; padding-bottom: 7rem; max-width: 56rem; }}

/* Display face earns its keep on titles only; everywhere else the sans carries the page. */
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2 {{
  font-family: var(--serif); font-weight: 400; letter-spacing: -0.015em;
  color: var(--text); margin-bottom: 0.15rem;
}}
[data-testid="stMarkdownContainer"] h1 {{ font-size: 2.6rem; line-height: 1.08; }}
[data-testid="stMarkdownContainer"] h2 {{ font-size: 1.9rem; line-height: 1.15; }}
[data-testid="stMarkdownContainer"] h3 {{
  font-family: var(--sans); font-size: 0.78rem; font-weight: 600;
  letter-spacing: 0.09em; text-transform: uppercase; color: var(--muted);
}}
[data-testid="stMarkdownContainer"] p {{ line-height: 1.62; }}

/* --- Shell ----------------------------------------------------------------------------- */
[data-testid="stSidebar"] {{ background: var(--ink); border-right: none; width: 17rem !important; }}
[data-testid="stSidebar"] * {{ color: #C9D4E0; }}
[data-testid="stSidebarUserContent"] {{ padding-top: 0.5rem; }}
[data-testid="stSidebarHeader"] {{ padding-bottom: 0; }}
[data-testid="stSidebarCollapseButton"] button {{ color: #7E8DA0 !important; }}

.ma-brand {{ display: flex; align-items: center; gap: 0.6rem; padding: 0.1rem 0 1.4rem; }}
.ma-brand-word {{
  font-family: var(--serif); font-size: 1.42rem; color: #F2F6FA;
  letter-spacing: -0.01em; line-height: 1;
}}

/* Custom nav: st.page_link styled as a quiet rail, active state carried by the halo. */
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {{
  border-radius: var(--r-sm); padding: 0.46rem 0.6rem; margin: 0.1rem 0;
  transition: background 130ms ease;
}}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {{ background: var(--ink-raised); }}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] p {{
  font-size: 0.88rem; font-weight: 500; color: #A9B7C6;
}}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] {{
  background: var(--ink-raised);
}}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] p {{ color: var(--halo); }}

.ma-rail-label {{
  font-family: var(--sans); font-size: 0.68rem; font-weight: 600; letter-spacing: 0.11em;
  text-transform: uppercase; color: #6B7B8E; margin: 1.5rem 0 0.4rem;
}}
.ma-rail-rule {{ height: 1px; background: var(--ink-line); border: 0; margin: 1.3rem 0; }}

.ma-user {{ display: flex; align-items: center; gap: 0.65rem; padding: 0.15rem 0; }}
.ma-user-dot {{
  width: 30px; height: 30px; border-radius: 9px; flex-shrink: 0;
  background: var(--ink-raised); border: 1px solid var(--ink-line);
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 0.76rem; color: var(--halo); text-transform: uppercase;
}}
.ma-user-name {{ font-size: 0.86rem; font-weight: 500; color: #E3EAF2; line-height: 1.2; }}
.ma-user-role {{ font-family: var(--mono); font-size: 0.68rem; color: #6B7B8E; text-transform: uppercase; }}

/* Sidebar buttons read as controls on ink, not as pale cutouts. */
[data-testid="stSidebar"] .stButton > button {{
  background: var(--ink-raised); color: #D6E0EA; border: 1px solid var(--ink-line);
  border-radius: var(--r-sm); font-size: 0.84rem; font-weight: 500;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
  border-color: var(--halo); color: var(--halo); background: var(--ink-raised);
}}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
  background: var(--ink-raised); border: 1px dashed var(--ink-line); border-radius: var(--r-md);
}}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {{
  background: transparent; border: 1px solid var(--ink-line); color: #C9D4E0;
}}

/* --- Header ---------------------------------------------------------------------------- */
.ma-head {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 1rem; }}
.ma-head-eyebrow {{
  font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--muted);
}}
.ma-head-sub {{ color: var(--muted); font-size: 0.92rem; margin-top: 0.35rem; max-width: 34rem; }}

/* --- Chat ------------------------------------------------------------------------------- */
[data-testid="stChatMessage"] {{
  background: transparent; border: 0; padding: 0.3rem 0 1.1rem; gap: 0.85rem;
}}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {{
  background-color: transparent !important; background-repeat: no-repeat;
  background-size: contain; background-position: center;
  width: 32px !important; height: 32px !important; border: 0 !important;
}}
[data-testid="stChatMessageAvatarUser"] > * ,
[data-testid="stChatMessageAvatarAssistant"] > * {{ display: none !important; }}
[data-testid="stChatMessageAvatarAssistant"] {{ background-image: url("{_svg_data_uri(_AVATAR_ASSISTANT)}"); }}
[data-testid="stChatMessageAvatarUser"] {{ background-image: url("{_svg_data_uri(_AVATAR_USER)}"); }}

.ma-card {{
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-lg);
  padding: 1.15rem 1.3rem; box-shadow: 0 1px 2px rgba(16,26,36,0.04);
}}
.ma-said {{ color: var(--text); line-height: 1.6; }}
.ma-file {{
  display: inline-flex; align-items: center; gap: 0.5rem;
  font-family: var(--mono); font-size: 0.82rem; color: var(--muted);
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 999px; padding: 0.3rem 0.8rem;
}}

/* --- Image mount ------------------------------------------------------------------------ */
/* The frame is a mount, not an object: no padding, hairline edge, the image reaches it. */
.ma-mount {{
  border-radius: var(--r-md); overflow: hidden; background: var(--ink);
  border: 1px solid rgba(14,22,33,0.10); box-shadow: 0 6px 18px -12px rgba(14,22,33,0.5);
  line-height: 0;
}}
.ma-mount img {{ width: 100%; display: block; }}
.ma-mount-label {{
  font-family: var(--mono); font-size: 0.66rem; letter-spacing: 0.09em;
  text-transform: uppercase; color: var(--muted); margin-top: 0.42rem; display: block;
}}

/* --- Data ------------------------------------------------------------------------------- */
.ma-badge {{
  display: inline-flex; align-items: center; gap: 0.42rem;
  padding: 0.24rem 0.66rem; margin: 0.16rem 0.3rem 0.16rem 0;
  border-radius: 999px; font-size: 0.8rem; font-weight: 500; letter-spacing: -0.005em;
}}
.ma-badge-score {{ font-family: var(--mono); font-size: 0.72rem; opacity: 0.82; }}
.ma-mono {{ font-family: var(--mono); font-size: 0.78rem; color: var(--muted); }}
.ma-caption {{ font-size: 0.78rem; color: var(--muted); line-height: 1.5; }}
.ma-divider {{ height: 1px; background: var(--line); border: 0; margin: 1.5rem 0; }}
.ma-flag-aligned {{ color: var(--muted); }}
.ma-flag-unaligned {{ color: var(--caution); }}

/* One finding, one card. Cards carry the separation that hairlines were doing badly, and
   they give the explanation a full-width line to run on instead of a narrow third column. */
[class*="st-key-finding_"] {{
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-lg);
  padding: 1.25rem 1.4rem; margin-bottom: 1.15rem;
  box-shadow: 0 1px 2px rgba(16,26,36,0.04);
  overflow-wrap: anywhere;
}}
/* Streamlit markdown paragraphs carry their own trailing margin, which escaped the card's
   bottom padding and left the last line sitting on the border. Spacing inside a card is the
   card's job, so the paragraphs give it up at the edges. */
[class*="st-key-finding_"] [data-testid="stMarkdownContainer"] p {{ margin-bottom: 0.6rem; }}
[class*="st-key-finding_"] [data-testid="stMarkdownContainer"] p:last-child {{ margin-bottom: 0; }}
[class*="st-key-finding_"] [data-testid="stMarkdownContainer"] :is(ul, ol) {{
  margin: 0.2rem 0 0.4rem 1.1rem;
}}
/* Below the point where three columns stop being three columns, the card should give its
   padding back to the content rather than keep a desktop-sized margin. */
@media (max-width: 640px) {{
  [class*="st-key-finding_"] {{ padding: 1rem 1rem; }}
  .ma-meter-name {{ width: 7rem; }}
}}
.ma-flag-chip {{
  display: inline-flex; align-items: center; gap: 0.42rem;
  font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.03em;
  padding: 0.28rem 0.66rem; border-radius: 999px;
  border: 1px solid var(--line); background: var(--film); color: var(--muted);
}}
.ma-flag-chip.is-atypical {{
  color: var(--caution); border-color: #E7D3AE; background: #FCF6EA;
}}
.ma-flag-dot {{ width: 5px; height: 5px; border-radius: 50%; background: currentColor; }}

/* History rows are buttons so the thread behind them can stay unfetched until opened, but
   they should read as list rows, not as controls competing for a click. */
[class*="st-key-histrow_"] {{ margin-bottom: 0.45rem; }}
[class*="st-key-histrow_"] .stButton > button {{
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-md);
  padding: 0.72rem 1rem; justify-content: flex-start; text-align: left;
  font-weight: 500; font-size: 0.89rem; color: var(--text);
}}
[class*="st-key-histrow_"] .stButton > button:hover {{
  border-color: var(--signal); background: var(--surface); color: var(--signal);
}}
/* Example questions are prompts to pick up, not controls competing with Ask: quiet, left
   aligned, no border until hovered. */
[class*="st-key-analytics_example_"] .stButton > button {{
  background: transparent; border: 1px solid transparent; color: var(--muted);
  justify-content: flex-start; text-align: left; padding: 0.32rem 0.55rem;
  font-size: 0.85rem; font-weight: 400;
}}
[class*="st-key-analytics_example_"] .stButton > button:hover {{
  background: var(--surface); border-color: var(--line); color: var(--signal);
}}
[class*="st-key-analytics_example_"] .stButton > button p {{ text-align: left; width: 100%; }}
[class*="st-key-analytics_example_"] {{ margin-bottom: -0.35rem; }}

[class*="st-key-histrow_"] .stButton > button p {{
  text-align: left; width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
/* The zone schematic is an instrument readout, so it sits on its own panel rather than
   floating as loose strokes on the page. */
.ma-zonecard {{
  border: 1px solid var(--line); border-radius: var(--r-md); background: var(--film);
  padding: 0.5rem 0.5rem 0.35rem; line-height: 0;
  display: block; width: 100%; max-width: 118px;
}}
.ma-zonecard svg {{ width: 100%; height: auto; display: block; }}

.ma-meter-row {{ display: flex; align-items: center; gap: 0.7rem; margin: 0.26rem 0; }}
.ma-meter-name {{ width: 9.6rem; flex-shrink: 0; font-size: 0.8rem; }}
.ma-meter-track {{ flex: 1; background: #E6EBF0; border-radius: 3px; height: 0.4rem; }}
.ma-meter-fill {{ height: 100%; border-radius: 3px; }}
.ma-meter-value {{ width: 2.7rem; text-align: right; font-family: var(--mono); font-size: 0.76rem; }}

/* --- Empty states ----------------------------------------------------------------------- */
.ma-empty {{
  position: relative; text-align: center; padding: 3.6rem 1rem 3rem;
}}
.ma-empty-art {{ display: flex; justify-content: center; margin-bottom: 1.1rem; }}
.ma-empty-title {{ font-family: var(--serif); font-size: 1.6rem; color: var(--text); }}
.ma-empty-body {{ color: var(--muted); font-size: 0.92rem; max-width: 27rem; margin: 0.4rem auto 0; line-height: 1.6; }}

/* --- Controls --------------------------------------------------------------------------- */
[data-testid="stMain"] .stButton > button {{
  background: var(--surface); color: var(--text); border: 1px solid var(--line);
  border-radius: var(--r-sm); font-size: 0.86rem; font-weight: 500; padding: 0.42rem 1rem;
  transition: border-color 130ms ease, color 130ms ease, background 130ms ease;
}}
[data-testid="stMain"] .stButton > button:hover {{
  border-color: var(--signal); color: var(--signal); background: var(--signal-soft);
}}
[data-testid="stChatInput"] {{
  border: 1px solid var(--line); border-radius: var(--r-md); background: var(--surface);
  box-shadow: 0 2px 10px -6px rgba(16,26,36,0.18);
}}
[data-testid="stTextInput"] input {{
  border-radius: var(--r-sm); border: 1px solid var(--line); background: var(--surface);
  font-family: var(--sans);
}}
[data-testid="stExpander"] {{
  border: 1px solid var(--line); border-radius: var(--r-md); background: var(--surface);
}}
[data-testid="stExpander"] summary p {{ font-size: 0.85rem; font-weight: 500; }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--line); border-radius: var(--r-md); }}
[data-testid="stAlertContainer"] {{ border-radius: var(--r-md); }}

/* Chrome we don't want, minus the parts that carry navigation. */
footer {{ visibility: hidden; }}
#MainMenu, [data-testid="stMainMenu"] {{ display: none; }}
[data-testid="stAppDeployButton"] {{ display: none; }}
[data-testid="stHeader"] {{ background: transparent; }}

@media (prefers-reduced-motion: reduce) {{
  * {{ transition: none !important; animation: none !important; }}
}}
</style>
"""


def inject_css() -> None:
    """Inject the stylesheet; call once per page render."""
    import streamlit as st
    st.markdown(_CSS, unsafe_allow_html=True)


def image_mount(url: str, label: str | None = None, max_width: str | None = None) -> str:
    """Return HTML for one image on its dark mount, wrapper and img in a single block.

    Streamlit isolates each st.markdown call, so an opening <div> emitted on its own is
    auto-closed and renders as an empty box above an unwrapped st.image. Emitting both
    together is the only way the mount actually holds the image.
    """
    style = f' style="max-width:{max_width}"' if max_width else ""
    caption = f'<span class="ma-mount-label">{label}</span>' if label else ""
    return f'<div{style}><div class="ma-mount"><img src="{url}"></div>{caption}</div>'


def condition_badge(condition: str, score: float | None = None) -> str:
    """Return HTML for a condition badge tinted with that condition's GradCAM hue."""
    colors = CONDITION_COLORS.get(condition, {"text": TEXT_MUTED, "bg": SIGNAL_SOFT})
    label = condition.replace("_", " ")
    score_html = f'<span class="ma-badge-score">{score:.2f}</span>' if score is not None else ""
    return (
        f'<span class="ma-badge" style="background:{colors["bg"]};color:{colors["text"]}">'
        f"{label}{score_html}</span>"
    )


def empty_state(title: str, body: str) -> str:
    """Return HTML for an empty state: an invitation to act, over the zone watermark."""
    return (
        f'<div class="ma-empty"><div class="ma-empty-art">{zone_watermark(150)}</div>'
        f'<div class="ma-empty-title">{title}</div>'
        f'<div class="ma-empty-body">{body}</div></div>'
    )