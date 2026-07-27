"""bg_remover.py – Background removal wrapper using withoutbg (WBGNet ONNX).

Uses the withoutBG Open Weights Model which runs the WBGNet ONNX graph locally.
Downloads the model from Hugging Face on first use (~small sidecar + ONNX graph),
then caches it. Returns a PIL Image in RGBA mode with the subject isolated on a
transparent background.

Usage
-----
    from comic_renderer.compositor.bg_remover import remove_background
    rgba = remove_background(pil_rgb_image)
"""

from __future__ import annotations

import logging

from PIL import Image

logger = logging.getLogger(__name__)

# Module-level singleton so the ONNX model is only loaded once per process
_model = None


def _get_model():
    """Lazily initialise and return the withoutBG OpenWeightsModel singleton."""
    global _model
    if _model is None:
        try:
            from withoutbg import OpenWeightsModel  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "withoutbg is required for background removal. "
                "Install it with: pip install withoutbg"
            ) from exc

        logger.debug("Loading withoutBG Open Weights Model (WBGNet ONNX)…")
        _model = OpenWeightsModel()
        # Eagerly preload so the first call to remove_background is fast
        _model.preload()
        logger.debug("withoutBG model loaded.")

    return _model


def remove_background(image: Image.Image) -> Image.Image:
    """Remove the background from *image* and return an RGBA PIL Image.

    Parameters
    ----------
    image:
        Source image in any PIL mode (RGB, RGBA, etc.).

    Returns
    -------
    PIL.Image.Image
        RGBA image where background pixels are transparent (alpha = 0)
        and the foreground subject retains its original colours.
    """
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    logger.debug("Removing background (withoutBG WBGNet ONNX)…")
    model = _get_model()
    result: Image.Image = model.remove_background(image)

    if result.mode != "RGBA":
        result = result.convert("RGBA")

    return result
