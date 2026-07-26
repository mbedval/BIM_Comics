"""GrayscaleFilter – converts an RGB image to grayscale (3-channel output).

The filter always returns a 3-channel ``(H, W, 3)`` uint8 array so the
downstream pipeline contract is not broken.  The three channels are identical
(R = G = B) and carry the grayscale luminance value.

Supported methods
-----------------
luminance (default)
    BT.601 weighted sum: ``L = 0.299·R + 0.587·G + 0.114·B``.
    Best perceptual match to human brightness perception.

average
    Arithmetic mean of the three channels: ``(R + G + B) / 3``.
    Simple but can over-lighten green and under-lighten blue.

lightness
    GIMP / HSL model: ``(max(R,G,B) + min(R,G,B)) / 2``.
    Preserves lightness as perceived in the HSL colour space.

JSON preset usage
-----------------
.. code-block:: json

    {"filter": "grayscale", "params": {"method": "luminance"}}
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from comic_renderer.filters.base import BaseFilter

_VALID_METHODS: frozenset[str] = frozenset({"luminance", "average", "lightness"})


class GrayscaleFilter(BaseFilter):
    """Convert an RGB image to grayscale, returning a 3-channel array.

    Parameters
    ----------
    params:
        ``method`` – one of ``"luminance"`` (default), ``"average"``,
        or ``"lightness"``.
    """

    FILTER_NAME: str = "grayscale"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._method: str = self._params.get("method", "luminance")
        if self._method not in _VALID_METHODS:
            raise ValueError(
                f"GrayscaleFilter: unknown method {self._method!r}. "
                f"Valid options: {sorted(_VALID_METHODS)}"
            )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Convert *image* to grayscale.

        Parameters
        ----------
        image:
            Input RGB ``uint8`` array ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Grayscale image replicated to 3 equal channels ``(H, W, 3)``
            dtype ``uint8``.
        """
        if self._method == "luminance":
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        elif self._method == "average":
            # Use float32 to avoid uint8 overflow during summation.
            gray = image.mean(axis=2).astype(np.uint8)

        else:  # lightness
            # HSL lightness = (max + min) / 2 per pixel.
            ch_max = image.max(axis=2).astype(np.uint16)
            ch_min = image.min(axis=2).astype(np.uint16)
            gray = ((ch_max + ch_min) // 2).astype(np.uint8)

        # Replicate single channel → 3 channels so shape is preserved.
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
