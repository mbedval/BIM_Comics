"""layout.py – Defines 4-panel asymmetric comic page geometry.

The page is divided into 4 panels in an asymmetric layout that mirrors
classic comic-book page design:

    ┌────────────────────┬──────────────┐
    │                    │              │
    │    Panel 0         │   Panel 1    │
    │    (58% width)     │  (42% width) │
    │                    │              │
    ├────────────┬───────┴──────────────┤
    │            │                      │
    │  Panel 2   │     Panel 3          │
    │ (42% width)│    (58% width)       │
    │            │                      │
    └────────────┴──────────────────────┘

Panel rectangles are returned as (x0, y0, x1, y1) tuples in pixel coords,
accounting for border width on all sides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

# Type alias for bounding box
BBox = Tuple[int, int, int, int]   # (x0, y0, x1, y1)


@dataclass(frozen=True)
class PageLayout:
    """Holds geometry of the full page and each of the 4 panels."""

    page_width: int
    page_height: int
    border: int
    panels: List[BBox]   # 4 entries, index matches panel number

    @property
    def page_size(self) -> Tuple[int, int]:
        return (self.page_width, self.page_height)

    def panel_size(self, index: int) -> Tuple[int, int]:
        x0, y0, x1, y1 = self.panels[index]
        return (x1 - x0, y1 - y0)


def build_layout(
    page_width: int = 1400,
    page_height: int = 1000,
    border: int = 8,
    row_split: float = 0.55,    # fraction of height for top row
    col_split_top: float = 0.58,  # fraction of width for top-left panel
    col_split_bot: float = 0.42,  # fraction of width for bottom-left panel
) -> PageLayout:
    """Compute panel bounding boxes for the 4-panel asymmetric layout.

    Parameters
    ----------
    page_width, page_height:
        Canvas dimensions in pixels.
    border:
        Width of the black divider / outer border in pixels.
    row_split:
        Vertical split point as a fraction of page height.
    col_split_top:
        Horizontal split for the TOP row (Panel 0 | Panel 1).
    col_split_bot:
        Horizontal split for the BOTTOM row (Panel 2 | Panel 3).
    """
    b = border
    W = page_width
    H = page_height

    # Row split pixel coordinate (centre of the horizontal divider)
    row_y = int(H * row_split)

    # Top-row column split
    top_col_x = int(W * col_split_top)

    # Bottom-row column split
    bot_col_x = int(W * col_split_bot)

    panels: List[BBox] = [
        # Panel 0 – top-left (large)
        (b,             b,          top_col_x - b,   row_y - b),
        # Panel 1 – top-right
        (top_col_x + b, b,          W - b,           row_y - b),
        # Panel 2 – bottom-left
        (b,             row_y + b,  bot_col_x - b,   H - b),
        # Panel 3 – bottom-right (large)
        (bot_col_x + b, row_y + b,  W - b,           H - b),
    ]

    return PageLayout(
        page_width=W,
        page_height=H,
        border=b,
        panels=panels,
    )
