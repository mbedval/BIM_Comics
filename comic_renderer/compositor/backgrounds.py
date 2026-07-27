"""backgrounds.py – Comic-style panel background generators using Pillow.

Provides three pattern generators:
- HalftoneBackground   – grid of dots on a solid colour base
- SunburstBackground   – radial lines emanating from a centre point
- DotsRadialBackground – dots that decrease in size toward the edges

Each generator returns a PIL Image (RGB) of the requested size.
"""

from __future__ import annotations

import math
from typing import Tuple

from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Color = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Panel colour palettes  (bg_color, pattern_color, pattern_type)
# ---------------------------------------------------------------------------
PANEL_STYLES: list[dict] = [
    {
        "bg_color": (255, 95, 162),       # Hot Pink
        "pattern_color": (220, 40, 110),  # Deep Pink
        "pattern": "halftone",
    },
    {
        "bg_color": (232, 48, 48),        # Vivid Red
        "pattern_color": (168, 24, 24),   # Dark Red
        "pattern": "halftone",
    },
    {
        "bg_color": (76, 209, 55),        # Lime Green
        "pattern_color": (39, 174, 96),   # Forest Green
        "pattern": "halftone",
    },
    {
        "bg_color": (14, 165, 233),       # Sky Blue
        "pattern_color": (0, 112, 192),   # Mid Blue
        "pattern": "sunburst",
    },
]


# ---------------------------------------------------------------------------
# Halftone background
# ---------------------------------------------------------------------------

def halftone_background(
    width: int,
    height: int,
    bg_color: Color,
    dot_color: Color,
    spacing: int = 28,
    dot_radius_fraction: float = 0.35,
) -> Image.Image:
    """Return a RGB PIL Image with a halftone dot grid pattern.

    Parameters
    ----------
    width, height:
        Output image dimensions in pixels.
    bg_color:
        Background fill colour (R, G, B).
    dot_color:
        Dot fill colour (R, G, B).
    spacing:
        Grid cell size in pixels. Dots are centred in each cell.
    dot_radius_fraction:
        Dot radius as a fraction of spacing (0 < f < 0.5).
    """
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    radius = int(spacing * dot_radius_fraction)
    # Offset every other row for a classic halftone stagger
    for row_idx, y in enumerate(range(0, height + spacing, spacing)):
        x_offset = spacing // 2 if row_idx % 2 else 0
        for x in range(-spacing + x_offset, width + spacing, spacing):
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=dot_color,
            )

    return img


# ---------------------------------------------------------------------------
# Sunburst background
# ---------------------------------------------------------------------------

def sunburst_background(
    width: int,
    height: int,
    bg_color: Color,
    ray_color: Color,
    num_rays: int = 24,
    cx_frac: float = 0.5,
    cy_frac: float = 0.5,
) -> Image.Image:
    """Return a RGB PIL Image with alternating sunburst rays.

    Parameters
    ----------
    width, height:
        Output image dimensions in pixels.
    bg_color:
        Colour of alternate rays (and background).
    ray_color:
        Colour of the primary rays.
    num_rays:
        Total number of rays (half will be bg_color, half ray_color).
    cx_frac, cy_frac:
        Fractional position of the burst centre (0–1).
    """
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    cx = width * cx_frac
    cy = height * cy_frac
    # Diagonal of the image guarantees rays reach all corners
    radius = math.hypot(width, height)

    angle_step = 2 * math.pi / num_rays

    for i in range(num_rays):
        if i % 2 == 0:
            continue  # bg_color rays are already the background fill
        angle_start = i * angle_step - angle_step / 2
        angle_end   = i * angle_step + angle_step / 2

        # Build polygon: centre + two far edge points
        pts = [
            (cx, cy),
            (cx + radius * math.cos(angle_start), cy + radius * math.sin(angle_start)),
            (cx + radius * math.cos(angle_end),   cy + radius * math.sin(angle_end)),
        ]
        draw.polygon(pts, fill=ray_color)

    return img


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def make_panel_background(
    width: int,
    height: int,
    style_index: int,
) -> Image.Image:
    """Return a comic-style panel background for the given style slot (0–3)."""
    style = PANEL_STYLES[style_index % len(PANEL_STYLES)]
    bg    = style["bg_color"]
    pat   = style["pattern_color"]

    if style["pattern"] == "sunburst":
        return sunburst_background(width, height, bg, pat)
    else:
        return halftone_background(width, height, bg, pat)
