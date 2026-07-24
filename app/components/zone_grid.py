"""Render the 7-zone anatomical grid as a compact SVG schematic.

Coordinates mirror CHEST_REGIONS in core/gradcam/region_map.py exactly, so the schematic
shows the same zone boundaries the backend actually measured activation against — this is
a readout of real data, not a decorative approximation.
"""

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

_STROKE = "#C6CDD4"
_STROKE_ACTIVE = "#2B4C5C"
_FILL_ACTIVE = "#2B4C5C1F"
_LABEL = "#8A94A0"


def zone_grid_svg(dominant_zones: list[str], width: int = 92, height: int = 116) -> str:
    """Return an SVG schematic of the 7 chest zones with the dominant ones highlighted."""
    active = set(dominant_zones or [])
    parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 100 126" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Chest zones: {", ".join(sorted(active)) or "none dominant"}">'
    ]

    for zone, (x1, y1, x2, y2) in _ZONES.items():
        is_active = zone in active
        px, py = x1 * 100, y1 * 126
        w, h = (x2 - x1) * 100, (y2 - y1) * 126
        parts.append(
            f'<rect x="{px:.1f}" y="{py:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{_FILL_ACTIVE if is_active else "none"}" '
            f'stroke="{_STROKE_ACTIVE if is_active else _STROKE}" '
            f'stroke-width="{1.2 if is_active else 0.7}" rx="1.5" />'
        )
        if is_active:
            parts.append(
                f'<text x="{px + w / 2:.1f}" y="{py + h / 2 + 2.5:.1f}" '
                f'text-anchor="middle" font-family="IBM Plex Mono, monospace" '
                f'font-size="7" fill="{_STROKE_ACTIVE}">{zone}</text>'
            )

    parts.append("</svg>")
    return "".join(parts)