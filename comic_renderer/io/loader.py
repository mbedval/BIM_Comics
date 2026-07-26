"""Image loader – reads images from disk into NumPy arrays.

Design decisions
----------------
* Uses OpenCV (``cv2.imread``) because it handles the widest range of formats
  with consistent behaviour across platforms.
* Converts BGR → RGB immediately so every downstream component works in RGB.
* Preserves the original bit-depth (``cv2.IMREAD_UNCHANGED`` flag is **not**
  used here because the pipeline operates on 8-bit images; EXIF orientation is
  corrected via Pillow when OpenCV fails to respect it).
* Does **not** resize or crop.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ExifTags

logger = logging.getLogger(__name__)


def _exif_rotation_code(pil_image: Image.Image) -> Optional[int]:
    """Return the EXIF orientation tag value, or *None* if absent."""
    try:
        exif_data = pil_image._getexif()  # type: ignore[attr-defined]
    except (AttributeError, Exception):
        return None

    if not exif_data:
        return None

    orientation_tag = next(
        (tag for tag, name in ExifTags.TAGS.items() if name == "Orientation"),
        None,
    )
    if orientation_tag is None:
        return None

    return exif_data.get(orientation_tag)


_EXIF_TRANSPOSE_MAP: dict[int, int] = {
    2: Image.FLIP_LEFT_RIGHT,
    3: Image.ROTATE_180,
    4: Image.FLIP_TOP_BOTTOM,
    5: Image.TRANSPOSE,
    6: Image.ROTATE_270,
    7: Image.TRANSVERSE,
    8: Image.ROTATE_90,
}


def load_image(path: Path) -> np.ndarray:
    """Load an image file into an RGB NumPy array.

    The loader applies EXIF orientation correction so that images shot in
    portrait mode on a phone are returned right-way-up regardless of the
    OpenCV version installed.

    Parameters
    ----------
    path:
        Absolute or relative path to the source image file.

    Returns
    -------
    np.ndarray
        Image array with shape ``(H, W, 3)`` and dtype ``uint8`` in RGB
        channel order.

    Raises
    ------
    FileNotFoundError
        When *path* does not exist.
    ValueError
        When OpenCV cannot decode the file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    logger.debug("Loading image: %s", path)

    # --- Primary load via OpenCV (BGR) ---
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"OpenCV could not decode image: {path}")

    # --- Convert BGR → RGB ---
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # --- EXIF orientation correction via Pillow ---
    try:
        pil_img = Image.open(path)
        code = _exif_rotation_code(pil_img)
        if code and code in _EXIF_TRANSPOSE_MAP:
            pil_img = pil_img.transpose(_EXIF_TRANSPOSE_MAP[code])
            rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
            logger.debug(
                "Applied EXIF orientation %d to: %s", code, path.name
            )
    except Exception as exc:  # noqa: BLE001
        # Pillow failure is non-fatal; we already have the OpenCV result.
        logger.warning(
            "EXIF orientation check failed for '%s': %s", path.name, exc
        )

    logger.debug(
        "Loaded '%s' → shape=%s dtype=%s", path.name, rgb.shape, rgb.dtype
    )
    return rgb
