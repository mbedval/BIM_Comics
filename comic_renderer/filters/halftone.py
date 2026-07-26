"""HalftoneFilter – converts an image into a binary or blended halftone dot pattern.

Simulates the classic newspaper print effect using rotated grid thresholds.
The grid cells are oriented at a screen angle, and pixel luminance is compared
against the radial distance from the cell center to create variable-sized dots.

JSON preset usage
-----------------
.. code-block:: json

    {
        "filter": "halftone",
        "params": {
            "size": 6,
            "angle": 45.0,
            "blend_strength": 0.8
        }
    }
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from comic_renderer.filters.base import BaseFilter


class HalftoneFilter(BaseFilter):
    """Apply a rotated-grid halftone screen to the image.

    Parameters
    ----------
    params:
        size : int, default 8
            Spacing between dot centers in pixels. Must be >= 2.
        angle : float, default 45.0
            Rotation angle of the halftone grid in degrees.
        blend_strength : float, default 0.8
            Weight to blend the halftone screen with the input image in [0, 1].
            1.0 produces a pure black-and-white print; lower values overlay it.
    """

    FILTER_NAME: str = "halftone"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)

        self._size: int = int(self._params.get("size", 8))
        if self._size < 2:
            raise ValueError(
                f"HalftoneFilter: size must be >= 2, got {self._size!r}"
            )

        self._angle: float = float(self._params.get("angle", 45.0))

        self._blend_strength: float = float(self._params.get("blend_strength", 0.8))
        if not (0.0 <= self._blend_strength <= 1.0):
            raise ValueError(
                f"HalftoneFilter: blend_strength must be in [0, 1], got {self._blend_strength!r}"
            )

        # Cache variables for shape-based performance optimizations
        self._cached_shape: tuple[int, int] | None = None
        self._cached_threshold: np.ndarray | None = None

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the halftone dot pattern to *image*.

        Parameters
        ----------
        image:
            Input RGB ``uint8`` array ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Halftoned RGB ``uint8`` array ``(H, W, 3)``.
        """
        if self._blend_strength == 0.0:
            return image.copy()

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if self._cached_shape != (h, w) or self._cached_threshold is None:
            self._cached_shape = (h, w)
            # Generate coordinate indices
            y_indices, x_indices = np.indices((h, w), dtype=np.float64)

            # Compute rotation trig values
            rad = self._angle * np.pi / 180.0
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)

            # Rotated grid coordinates
            xr = x_indices * cos_a - y_indices * sin_a
            yr = x_indices * sin_a + y_indices * cos_a

            # Local grid coordinates within size x size cell
            px = np.mod(xr, self._size)
            py = np.mod(yr, self._size)

            # Shift center of cell to (size/2, size/2)
            half_size = self._size / 2.0
            dx = px - half_size
            dy = py - half_size
            d = np.sqrt(dx**2 + dy**2)

            # Maximum possible distance from center of cell
            d_max = self._size * np.sqrt(2.0) / 2.0
            d_norm = d / d_max

            # Threshold grid: 255 at corners (easy to exceed), 0 at center (never exceeded)
            self._cached_threshold = 255.0 * (1.0 - d_norm)

        # Convert to halftone binary (black dot if gray <= threshold)
        halftone_gray = np.where(gray.astype(np.float64) > self._cached_threshold, 255, 0).astype(np.uint8)

        # Convert single-channel halftone to 3-channel RGB
        halftone_rgb = cv2.cvtColor(halftone_gray, cv2.COLOR_GRAY2RGB)

        # Blend original image with halftone pattern
        result = image.astype(np.float64) * (1.0 - self._blend_strength) + halftone_rgb.astype(np.float64) * self._blend_strength
        return np.clip(result, 0.0, 255.0).astype(np.uint8)
