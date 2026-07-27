"""page_builder.py – Assembles a comic page from 4 images.

Workflow for each panel:
  1. Load source image from disk.
  2. Apply the chosen comic preset pipeline (e.g. cartoon, anime, noir).
  3. Remove the background → RGBA with transparent BG.
  4. Generate the colourful comic panel background (halftone / sunburst).
  5. Composite the subject (RGBA) onto the background.
  6. Paste the panel onto the master page canvas at the correct position.

After all 4 panels are placed, draw the thick black borders, then save.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

from comic_renderer.compositor.backgrounds import make_panel_background
from comic_renderer.compositor.bg_remover import remove_background
from comic_renderer.compositor.layout import PageLayout, build_layout

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_rgb(path: Path) -> np.ndarray:
    """Load an image from disk as a numpy RGB uint8 array."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _apply_preset(image: np.ndarray, preset_name: str) -> np.ndarray:
    """Run *image* through the named preset pipeline and return the result."""
    # Import here to avoid circular imports at module load time
    from comic_renderer.bis_comic_main import _build_registry   # type: ignore[import]
    from comic_renderer.pipeline.executor import PipelineExecutor
    from comic_renderer.pipeline.preset_loader import PresetLoader

    _PRESETS_DIR = Path(__file__).parent.parent / "presets"

    registry = _build_registry()
    loader = PresetLoader(_PRESETS_DIR)
    preset_cfg = loader.load(preset_name)
    executor = PipelineExecutor(registry)
    return executor.run(image, preset_cfg)


def _fit_subject_in_panel(
    subject_rgba: Image.Image,
    panel_width: int,
    panel_height: int,
    padding_fraction: float = 0.05,
) -> tuple[Image.Image, tuple[int, int]]:
    """Scale *subject_rgba* to fit inside the panel, preserving aspect ratio.

    Returns the scaled image and the (x, y) paste offset so the subject is
    centred in the panel.
    """
    pad_x = int(panel_width  * padding_fraction)
    pad_y = int(panel_height * padding_fraction)
    max_w = panel_width  - 2 * pad_x
    max_h = panel_height - 2 * pad_y

    sw, sh = subject_rgba.size
    scale = min(max_w / sw, max_h / sh)
    new_w = int(sw * scale)
    new_h = int(sh * scale)

    scaled = subject_rgba.resize((new_w, new_h), Image.LANCZOS)

    # Centre inside the panel
    offset_x = (panel_width  - new_w) // 2
    offset_y = (panel_height - new_h) // 2
    return scaled, (offset_x, offset_y)


def _draw_borders(canvas: Image.Image, layout: PageLayout) -> None:
    """Draw thick black outer border and inner dividers on *canvas* in-place."""
    draw = ImageDraw.Draw(canvas)
    b = layout.border
    W, H = layout.page_size

    # Outer frame
    draw.rectangle([0, 0, W - 1, H - 1], outline=(0, 0, 0), width=b)

    # Find the approximate horizontal divider Y from panel geometry
    # (bottom of Panel 0, top of Panel 2)
    _, _, _, row_y_bottom = layout.panels[0]
    _, row_y_top, _, _   = layout.panels[2]
    mid_y = (row_y_bottom + row_y_top) // 2
    draw.rectangle([0, mid_y - b // 2, W, mid_y + b // 2], fill=(0, 0, 0))

    # Vertical divider – top row
    x0_p1, _, _, _ = layout.panels[1]
    top_divider_x = (layout.panels[0][2] + x0_p1) // 2
    draw.rectangle([top_divider_x - b // 2, 0, top_divider_x + b // 2, mid_y], fill=(0, 0, 0))

    # Vertical divider – bottom row
    x0_p3, _, _, _ = layout.panels[3]
    bot_divider_x = (layout.panels[2][2] + x0_p3) // 2
    draw.rectangle([bot_divider_x - b // 2, mid_y, bot_divider_x + b // 2, H], fill=(0, 0, 0))


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

class ComicPageBuilder:
    """Build a single comic page from 4 source images.

    Parameters
    ----------
    preset_name:
        Name of a registered preset to apply to each source image.
        Use ``None`` to skip the pipeline step (raw images only).
    page_width, page_height:
        Canvas size in pixels.
    border:
        Thickness of black dividers / outer frame in pixels.
    skip_bg_removal:
        If True, skip background removal (useful for testing / speed).
    """

    def __init__(
        self,
        preset_name: Optional[str] = "cartoon",
        page_width: int = 1400,
        page_height: int = 1000,
        border: int = 8,
        skip_bg_removal: bool = False,
    ) -> None:
        self._preset_name    = preset_name
        self._skip_bg_removal = skip_bg_removal
        self._layout: PageLayout = build_layout(
            page_width=page_width,
            page_height=page_height,
            border=border,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, image_paths: List[Path], output_path: Path) -> None:
        """Assemble a comic page from *image_paths* (exactly 4) and save it.

        Parameters
        ----------
        image_paths:
            Ordered list of exactly 4 source image paths.
        output_path:
            Destination file path. The parent directory is created if needed.
        """
        if len(image_paths) != 4:
            raise ValueError(
                f"ComicPageBuilder requires exactly 4 images, got {len(image_paths)}."
            )

        layout  = self._layout
        canvas  = Image.new("RGB", layout.page_size, (0, 0, 0))

        for idx, img_path in enumerate(image_paths):
            logger.info("  Panel %d: processing '%s'…", idx, img_path.name)

            # 1. Load source
            rgb_arr = _load_rgb(img_path)

            # 2. Apply preset pipeline
            if self._preset_name:
                logger.info("    Applying preset '%s'…", self._preset_name)
                rgb_arr = _apply_preset(rgb_arr, self._preset_name)

            # 3. Remove background
            pil_rgb = Image.fromarray(rgb_arr.astype(np.uint8), mode="RGB")
            if self._skip_bg_removal:
                subject_rgba = pil_rgb.convert("RGBA")
            else:
                logger.info("    Removing background…")
                subject_rgba = remove_background(pil_rgb)

            # 4. Generate panel background
            x0, y0, x1, y1 = layout.panels[idx]
            pw, ph = x1 - x0, y1 - y0
            bg = make_panel_background(pw, ph, style_index=idx)

            # 5. Composite subject onto background
            panel_canvas = bg.convert("RGBA")
            scaled_subject, (sx, sy) = _fit_subject_in_panel(subject_rgba, pw, ph)
            panel_canvas.paste(scaled_subject, (sx, sy), mask=scaled_subject)

            # 6. Paste panel onto master canvas
            canvas.paste(panel_canvas.convert("RGB"), (x0, y0))
            logger.info("    ✓ Panel %d done.", idx)

        # Draw borders last so they sit on top
        _draw_borders(canvas, layout)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(str(output_path))
        logger.info("Comic page saved → %s", output_path)
