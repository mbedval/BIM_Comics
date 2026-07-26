"""PosterizeFilter – reduces the number of distinct colour levels per channel.

Posterization maps each 8-bit channel (0–255) into a small set of discrete
levels, producing the flat-colour, pop-art look typical of comic colouring
and screen-print reproduction.

Algorithm
---------
Given ``levels`` L, each channel is quantized into L equal-width buckets:

    step  = 256 // L
    value = (pixel // step) * step

Example with L=4 (step=64):

    pixel =   0  →  0
    pixel =  63  →  0
    pixel =  64  → 64
    pixel = 127  → 64
    pixel = 128  → 128
    pixel = 191  → 128
    pixel = 192  → 192
    pixel = 255  → 192

JSON preset usage
-----------------
.. code-block:: json

    {"filter": "posterize", "params": {"levels": 4}}
"""

from __future__ import annotations

from typing import Any

import numpy as np

from comic_renderer.filters.base import BaseFilter

_MIN_LEVELS: int = 2
_MAX_LEVELS: int = 256


class PosterizeFilter(BaseFilter):
    """Quantize each colour channel into a fixed number of distinct levels.

    Parameters
    ----------
    params:
        levels : int, default 4
            Number of output levels per channel.  Must be in [2, 256].
            Lower values produce a more stylised / posterised look.
    """

    FILTER_NAME: str = "posterize"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._levels: int = int(self._params.get("levels", 4))
        if not (_MIN_LEVELS <= self._levels <= _MAX_LEVELS):
            raise ValueError(
                f"PosterizeFilter: levels must be in [{_MIN_LEVELS}, {_MAX_LEVELS}], "
                f"got {self._levels!r}"
            )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Posterize *image* to ``self._levels`` levels per channel.

        Parameters
        ----------
        image:
            Input RGB ``uint8`` array ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Posterized RGB ``uint8`` array ``(H, W, 3)``.
        """
        step: int = 256 // self._levels

        # Promote to uint16 to avoid overflow when multiplying, then clip
        # back to [0, 255] before casting to uint8.
        quantized = (image.astype(np.uint16) // step * step)
        return np.clip(quantized, 0, 255).astype(np.uint8)
