"""Render the 7-zone anatomical grid as a compact SVG schematic."""

from app.theme import SIGNAL, TEXT_MUTED

# Normalized (x1, y1, x2, y2), matching core/gradcam/region_map.py CHEST_REGIONS
_ZONES: dict[str, tuple[float, float, float, float]] = {
    "RUZ": (0.00, 0.00, 0.50, 0.33),
    "LUZ": (0.50, 0.00, 1.00, 0.33),
    "RMZ": (0.00, 0.33, 0.42, 0.63),
    "LMZ": (0.58, 0.33, 1.00, 0.63),
    "CAR": (0.33, 0.30, 0.67, 0.73),
    "RLZ": (0.00, 0.63, 0.50, 1.00),
    "LLZ": (0.50, 0.63, 1.00, 1.00),
}

# Strong enough to read as a diagram on the film surface. The previous hairline sat at the
# threshold of visibility and the whole figure looked unfinished because of it.
_STROKE_IDLE = "#B7C2CE"
_LABEL_IDLE = "#93A0AE"
_FILL_ACTIVE = f"{SIGNAL}26"


def zone_grid_svg(dominant_zones: list[str], max_width: int = 104) -> str:
    """Return an SVG schematic of the 7 chest zones with the dominant ones highlighted.

    Sized by viewBox rather than fixed pixels so the figure shrinks with its column instead
    of forcing the column to stay wide enough for it.
    """
    active = set(dominant_zones or [])
    label = ", ".join(sorted(active)) or "none dominant"
    parts = [
        f'<svg viewBox="0 0 100 126" preserveAspectRatio="xMidYMid meet" '
        f'style="width:100%;height:auto;max-width:{max_width}px" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Chest zones, dominant: {label}">'
    ]

    for zone, (x1, y1, x2, y2) in _ZONES.items():
        is_active = zone in active
        px, py = x1 * 100, y1 * 126
        w, h = (x2 - x1) * 100, (y2 - y1) * 126
        parts.append(
            f'<rect x="{px:.1f}" y="{py:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{_FILL_ACTIVE if is_active else "none"}" '
            f'stroke="{SIGNAL if is_active else _STROKE_IDLE}" '
            f'stroke-width="{1.7 if is_active else 1.0}" rx="2.5" />'
        )
        parts.append(
            f'<text x="{px + w / 2:.1f}" y="{py + h / 2 + 2.6:.1f}" text-anchor="middle" '
            f'font-family="JetBrains Mono, monospace" font-size="7.5" letter-spacing="0.3" '
            f'font-weight="{500 if is_active else 400}" '
            f'fill="{SIGNAL if is_active else _LABEL_IDLE}">{zone}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def zone_panel(dominant_zones: list[str]) -> str:
    """Return the schematic mounted on its panel, with the dominant zones named beneath it."""
    named = ", ".join(dominant_zones) if dominant_zones else "none"
    return (
        f'<div class="ma-zonecard">{zone_grid_svg(dominant_zones)}</div>'
        f'<span class="ma-mount-label">dominant · {named}</span>'
    )