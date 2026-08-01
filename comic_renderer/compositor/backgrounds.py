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
        "bg_color": (255, 200, 220),       # Light Vibrant Pink
        "pattern_color": (255, 110, 160),  # Hot Pink
        "pattern": "halftone",
    },
    {
        "bg_color": (255, 245, 180),       # Light Creamy Gold
        "pattern_color": (255, 195, 0),    # Vibrant Yellow Gold
        "pattern": "sunburst",
    },
    {
        "bg_color": (210, 250, 210),       # Light Neon Mint
        "pattern_color": (46, 204, 113),   # Vibrant Emerald Green
        "pattern": "halftone",
    },
    {
        "bg_color": (200, 235, 255),       # Light Sky Blue
        "pattern_color": (52, 152, 219),   # Vibrant Dodger Blue
        "pattern": "halftone",
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
    shape: str = "circle",
) -> Image.Image:
    """Return a RGB PIL Image with a halftone grid pattern of varying shapes, super-sampled for crisp edges.

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
    shape:
        Pattern shape ("circle", "square", "triangle", "diamond").
    """
    factor = 3
    ss_width = width * factor
    ss_height = height * factor
    ss_spacing = spacing * factor

    img = Image.new("RGB", (ss_width, ss_height), bg_color)
    draw = ImageDraw.Draw(img)

    radius = int(ss_spacing * dot_radius_fraction)
    # Offset every other row for a classic halftone stagger
    for row_idx, y in enumerate(range(0, ss_height + ss_spacing, ss_spacing)):
        x_offset = ss_spacing // 2 if row_idx % 2 else 0
        for x in range(-ss_spacing + x_offset, ss_width + ss_spacing, ss_spacing):
            if shape == "square":
                draw.rectangle([x - radius, y - radius, x + radius, y + radius], fill=dot_color)
            elif shape == "triangle":
                h_offset = int(radius * 0.866)
                p1 = (x, y - radius)
                p2 = (x - h_offset, y + radius // 2)
                p3 = (x + h_offset, y + radius // 2)
                draw.polygon([p1, p2, p3], fill=dot_color)
            elif shape == "diamond":
                p1 = (x, y - radius)
                p2 = (x - radius, y)
                p3 = (x, y + radius)
                p4 = (x + radius, y)
                draw.polygon([p1, p2, p3, p4], fill=dot_color)
            else: # "circle"
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    fill=dot_color,
                )

    return img.resize((width, height), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# Sunburst background
# ---------------------------------------------------------------------------

def sunburst_background(
    width: int,
    height: int,
    bg_color: Color,
    ray_color: Color,
    num_rays: int = 24,
    cx_frac: float = -0.1,
    cy_frac: float = 1.1,
) -> Image.Image:
    """Return a RGB PIL Image with swept-diagonal rays, super-sampled for high anti-aliasing.

    Parameters
    ----------
    width, height:
        Output image dimensions in pixels.
    bg_color:
        Colour of alternate rays (and background).
    ray_color:
        Colour of the primary rays.
    num_rays:
        Total number of rays.
    cx_frac, cy_frac:
        Position of ray origin, defaults to off-screen bottom-left for a swept look.
    """
    factor = 3
    ss_width = width * factor
    ss_height = height * factor

    img = Image.new("RGB", (ss_width, ss_height), bg_color)
    draw = ImageDraw.Draw(img)

    cx = ss_width * cx_frac
    cy = ss_height * cy_frac
    radius = math.hypot(ss_width, ss_height) * 1.5

    angle_step = 2 * math.pi / num_rays

    for i in range(num_rays):
        if i % 2 == 0:
            continue
        angle_start = i * angle_step - angle_step / 2
        angle_end   = i * angle_step + angle_step / 2

        pts = [
            (cx, cy),
            (cx + radius * math.cos(angle_start), cy + radius * math.sin(angle_start)),
            (cx + radius * math.cos(angle_end),   cy + radius * math.sin(angle_end)),
        ]
        draw.polygon(pts, fill=ray_color)

    return img.resize((width, height), Image.Resampling.LANCZOS)


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
        shapes = ["circle", "square", "triangle", "square"]
        shape = shapes[style_index % len(shapes)]
        return halftone_background(width, height, bg, pat, shape=shape)
