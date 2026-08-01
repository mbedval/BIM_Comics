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

from comic_renderer.compositor.backgrounds import make_panel_background, halftone_background, sunburst_background
from comic_renderer.compositor.bg_remover import remove_background
from comic_renderer.compositor.layout import PageLayout, build_layout

logger = logging.getLogger(__name__)

# Global cache to reuse background removal across multiple presets (speeds up `--preset all` runs)
_BG_REMOVAL_CACHE: dict[Path, Image.Image] = {}


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


def _crop_subject_bounds(image: Image.Image) -> Image.Image:
    """Crop image to its active bounding box (removing empty transparent margins)."""
    bbox = image.getbbox()
    if bbox:
        logger.info("    Cropping subject margins to bounding box: %s", bbox)
        return image.crop(bbox)
    return image


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
    scale = max(max_w / sw, max_h / sh)
    new_w = int(sw * scale)
    new_h = int(sh * scale)

    scaled = subject_rgba.resize((new_w, new_h), Image.LANCZOS)

    # Centre inside the panel
    offset_x = (panel_width  - new_w) // 2
    offset_y = (panel_height - new_h) // 2
    return scaled, (offset_x, offset_y)


def _is_image_grayscale(pil_image: Image.Image) -> bool:
    """Check if the image is grayscale or near-grayscale (mean absolute difference between channels is low)."""
    if pil_image.mode in ("L", "1"):
        return True
    img = pil_image.convert("RGB")
    small = img.resize((15, 15), Image.Resampling.BOX)
    pixels = list(small.getdata())
    for r, g, b in pixels:
        if max(r, g, b) - min(r, g, b) > 18:
            return False
    return True


def _make_bw_style(idx: int) -> tuple[tuple[int, int, int], tuple[int, int, int], str]:
    """Return a light grayish color palette and pattern type for B&W presets."""
    pattern = "sunburst" if idx == 1 else "halftone"
    bg_shades = [(245, 245, 245), (240, 240, 240), (250, 250, 250), (235, 235, 235)]
    pat_shades = [(210, 210, 210), (190, 190, 190), (220, 220, 220), (170, 170, 170)]
    return bg_shades[idx % 4], pat_shades[idx % 4], pattern


def _draw_borders(canvas: Image.Image, layout: PageLayout, active_count: int) -> None:
    """Draw borders (no-op for sticker layout)."""
    pass


def _find_comic_font(size: int = 40):
    """Attempt to load a bold sans-serif or comic font from system paths, falling back to default."""
    from PIL import ImageFont
    paths = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except IOError:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _draw_to_be_continued(canvas: Image.Image, poly: list[tuple[int, int]], is_bw: bool) -> None:
    """Draw a stylized comic 'To be Continued....' sticker banner centered inside the polygon bounds."""
    # Compute polygon bounding box to center the banner
    xs = [pt[0] for pt in poly]
    ys = [pt[1] for pt in poly]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    pw, ph = x1 - x0, y1 - y0

    overlay = Image.new("RGBA", (canvas.width, canvas.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    bw, bh = int(pw * 0.70), int(ph * 0.22)
    bx0, by0 = x0 + (pw - bw) // 2, y0 + (ph - bh) // 2
    bx1, by1 = bx0 + bw, by0 + bh
    
    # Draw white banner body with black outline
    draw.rectangle([bx0, by0, bx1, by1], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=4)
    
    # Load bold font
    font = _find_comic_font(32)
    text = "To be Continued...."
    
    # Center text
    if hasattr(draw, "textbbox"):
        tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), text, font=font)
        tw, th = tx1 - tx0, ty1 - ty0
    else:
        tw, th = draw.textsize(text, font=font)
        
    tx = bx0 + (bw - tw) // 2
    ty = by0 + (bh - th) // 2
    
    # Draw text outline for legibility
    for ox, oy in [(-2, -2), (2, -2), (-2, 2), (2, 2), (-2, 0), (2, 0), (0, -2), (0, 2)]:
        draw.text((tx + ox, ty + oy), text, fill=(0, 0, 0, 255), font=font)
        
    # Yellow comic-style pop text for color pages, dark charcoal text for B&W pages
    text_color = (255, 215, 0, 255) if not is_bw else (60, 60, 60, 255)
    draw.text((tx, ty), text, fill=text_color, font=font)
    
    # Tilt the sticker slightly for a realistic hand-pasted look
    rotated = overlay.rotate(3, resample=Image.Resampling.BICUBIC, center=(bx0 + bw//2, by0 + bh//2))
    canvas.paste(rotated, (0, 0), mask=rotated)


def get_instagram_layout(W: int, H: int, b: int, layout_type: int) -> list[dict]:
    """Generate 2-panel polygon coordinates ensuring vertical parallel outer side edges."""
    if layout_type == 0:
        # Rising diagonal divider
        poly0 = [(20, 20), (W-20, 20), (W-20, 1020), (20, 900)]
        poly1 = [(20, 920), (W-20, 1040), (W-20, H-20), (20, H-20)]
    elif layout_type == 1:
        # Falling diagonal divider
        poly0 = [(20, 20), (W-20, 20), (W-20, 900), (20, 1020)]
        poly1 = [(20, 1040), (W-20, 920), (W-20, H-20), (20, H-20)]
    else:
        # Steep rising diagonal divider
        poly0 = [(20, 20), (W-20, 20), (W-20, 1080), (20, 840)]
        poly1 = [(20, 860), (W-20, 1100), (W-20, H-20), (20, H-20)]

    def get_bbox(poly):
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        return min(xs), min(ys), max(xs), max(ys)

    return [
        {"polygon": poly0, "bbox": get_bbox(poly0)},
        {"polygon": poly1, "bbox": get_bbox(poly1)},
    ]


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

class ComicPageBuilder:
    """Build one or more comic pages from source images.

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
        page_width: int = 1080,
        page_height: int = 1920,
        border: int = 8,
        skip_bg_removal: bool = False,
    ) -> None:
        self._preset_name    = preset_name
        self._skip_bg_removal = skip_bg_removal
        self._page_width     = page_width
        self._page_height    = page_height
        self._border         = border

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, image_paths: List[Path], output_path: Path) -> None:
        """Assemble comic page(s) from *image_paths* (any length) and save.

        Creates 2-panel layout pages of 1080x1920 dimensions.

        Parameters
        ----------
        image_paths:
            Ordered list of source image paths.
        output_path:
            Destination file path. The parent directory is created if needed.
        """
        if not image_paths:
            raise ValueError("ComicPageBuilder requires at least 1 image path.")

        # Chunk the images into pages of 2 for Instagram layout
        chunks = [image_paths[i : i + 2] for i in range(0, len(image_paths), 2)]
        total_pages = len(chunks)

        for page_idx, chunk in enumerate(chunks, 1):
            logger.info("Generating Page %d/%d with %d panel(s)…", page_idx, total_pages, len(chunk))
            W, H = self._page_width, self._page_height
            b = self._border

            # Determine color vs B&W theme from first processed image on the page
            first_rgb = _load_rgb(chunk[0])
            if self._preset_name:
                first_rgb = _apply_preset(first_rgb, self._preset_name)
            first_pil = Image.fromarray(first_rgb.astype(np.uint8), mode="RGB")
            page_is_bw = _is_image_grayscale(first_pil)

            # Generate 24 style permutations and choose one randomly
            import itertools
            import random
            perms = list(itertools.permutations([0, 1, 2, 3]))
            style_indices = list(random.choice(perms))

            # Select layout slice cut randomly (Trapezoids vs Parallelograms)
            layout_type = random.randint(0, 2)
            panels = get_instagram_layout(W, H, b, layout_type)

            # Page background:
            # - If 2 panels: use a clean neutral off-white paper background.
            # - If 1 panel: use the style of the empty panel as the page background.
            active_count = len(chunk)
            if active_count < 2:
                bg_idx = style_indices[1]  # Empty Panel 1 style index
                if page_is_bw:
                    page_bg, page_pat, page_pattern = _make_bw_style(bg_idx)
                else:
                    from comic_renderer.compositor.backgrounds import PANEL_STYLES
                    style = PANEL_STYLES[bg_idx]
                    page_bg, page_pat = style["bg_color"], style["pattern_color"]
                    page_pattern = style["pattern"]

                if page_pattern == "sunburst":
                    canvas = sunburst_background(W, H, page_bg, page_pat, num_rays=24)
                else:
                    shapes = ["circle", "square", "triangle", "diamond"]
                    shape = shapes[bg_idx % len(shapes)]
                    canvas = halftone_background(W, H, page_bg, page_pat, spacing=28, shape=shape)
            else:
                paper_color = (255, 255, 255) # Clean white gutter background
                canvas = Image.new("RGB", (W, H), paper_color)

            # Draw panels as custom stickers pasted using polygon masks
            for idx, img_path in enumerate(chunk):
                logger.info("  Panel %d: processing '%s'…", idx, img_path.name)

                # 1. Load and process source
                rgb_arr = _load_rgb(img_path)
                if self._preset_name:
                    logger.info("    Applying preset '%s'…", self._preset_name)
                    rgb_arr = _apply_preset(rgb_arr, self._preset_name)

                # 2. Convert to PIL and detect grayscale vs color
                pil_rgb = Image.fromarray(rgb_arr.astype(np.uint8), mode="RGB")
                panel_is_bw = _is_image_grayscale(pil_rgb)

                # 3. Remove background & crop
                if self._skip_bg_removal:
                    subject_rgba = pil_rgb.convert("RGBA")
                else:
                    if img_path not in _BG_REMOVAL_CACHE:
                        logger.info("    Removing background (first time for '%s')…", img_path.name)
                        # Load original raw image to ensure we compute background removal on unstylized source
                        raw_rgb = _load_rgb(img_path)
                        raw_pil = Image.fromarray(raw_rgb.astype(np.uint8), mode="RGB")
                        _BG_REMOVAL_CACHE[img_path] = remove_background(raw_pil)
                    
                    # Combine the stylized RGB image with the precomputed alpha mask
                    raw_rgba = _BG_REMOVAL_CACHE[img_path]
                    alpha_mask = raw_rgba.split()[3]
                    subject_rgba = Image.merge("RGBA", pil_rgb.split() + (alpha_mask,))
                    subject_rgba = _crop_subject_bounds(subject_rgba)

                # 4. Generate panel dimensions and dynamic background design
                panel_info = panels[idx]
                poly = panel_info["polygon"]
                px0, py0, px1, py1 = panel_info["bbox"]
                pw, ph = px1 - px0, py1 - py0
                
                style_idx = style_indices[idx]

                if panel_is_bw:
                    panel_bg, panel_pat, panel_pattern = _make_bw_style(style_idx)
                else:
                    from comic_renderer.compositor.backgrounds import PANEL_STYLES
                    style = PANEL_STYLES[style_idx]
                    panel_bg, panel_pat = style["bg_color"], style["pattern_color"]
                    panel_pattern = style["pattern"]

                dynamic_spacing = 18 + (style_idx * 8)
                
                shapes = ["circle", "square", "triangle", "diamond"]
                shape = shapes[style_idx % len(shapes)]

                if panel_pattern == "sunburst":
                    bg = sunburst_background(pw, ph, panel_bg, panel_pat, num_rays=12 + (style_idx * 4))
                else:
                    bg = halftone_background(pw, ph, panel_bg, panel_pat, spacing=dynamic_spacing, shape=shape)

                # 5. Composite subject onto panel background bounding box canvas (use 0% padding to cover panel border to border)
                panel_canvas = bg.convert("RGBA")
                scaled_subject, (sx, sy) = _fit_subject_in_panel(subject_rgba, pw, ph, padding_fraction=0.0)
                panel_canvas.paste(scaled_subject, (sx, sy), mask=scaled_subject)

                # 6. Polygon Mask: Create mask for this panel's specific cut coordinates
                mask_img = Image.new("L", (W, H), 0)
                mask_draw = ImageDraw.Draw(mask_img)
                mask_draw.polygon(poly, fill=255)

                # 7. Paste panel cut onto master canvas using polygon mask (no drop shadows)
                panel_full = Image.new("RGB", (W, H), (0, 0, 0))
                panel_full.paste(panel_canvas.convert("RGB"), (px0, py0))
                canvas.paste(panel_full, (0, 0), mask=mask_img)

                # 8. Black Frame Borders: Draw solid black outline around the cut polygon on the master canvas
                draw_canvas = ImageDraw.Draw(canvas)
                draw_canvas.polygon(poly, outline=(0, 0, 0), width=6)
                logger.info("    ✓ Panel %d done.", idx)

            # 9. Draw outlines for all panels (including empty panels) for a clean visual frame
            draw_canvas = ImageDraw.Draw(canvas)
            for panel_info in panels:
                draw_canvas.polygon(panel_info["polygon"], outline=(0, 0, 0), width=6)

            # 10. Draw a thin black border around the entire canvas for a clean finish
            draw_canvas.rectangle([0, 0, W - 1, H - 1], outline=(0, 0, 0), width=6)

            # Determine the output filename for this page
            if total_pages > 1:
                page_output_path = output_path.with_name(
                    f"{output_path.stem}_page{page_idx}{output_path.suffix}"
                )
            else:
                page_output_path = output_path

            # Save
            page_output_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(str(page_output_path))
            logger.info("Comic page saved → %s", page_output_path)
