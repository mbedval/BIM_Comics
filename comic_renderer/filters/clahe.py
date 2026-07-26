"""CLAHEFilter – Contrast Limited Adaptive Histogram Equalization.

CLAHE prevents the over-amplification of noise that plain histogram
equalization causes by clipping the histogram at a configurable limit before
computing the CDF.  The image is divided into a grid of non-overlapping tiles;
each tile's histogram is equalized independently and then the tiles are
bilinearly interpolated to produce a smooth result.

Colour handling
---------------
For colour (RGB) images the filter converts to CIE L*a*b* colour space,
applies CLAHE only to the L* (lightness) channel, and converts back.
This prevents colour shift while still enhancing local contrast.

For grayscale inputs (all three channels identical) the same LAB path is
used; the a* and b* channels remain near-zero so the output is still
effectively grayscale.

JSON preset usage
-----------------
.. code-block:: json

    {
        "filter": "clahe",
        "params": {
            "clip_limit": 2.0,
            "tile_grid_size": [8, 8]
        }
    }
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from comic_renderer.filters.base import BaseFilter

_MIN_TILE_SIZE: int = 1
_MIN_CLIP_LIMIT: float = 0.0  # exclusive lower bound


class CLAHEFilter(BaseFilter):
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Parameters
    ----------
    params:
        clip_limit : float, default 2.0
            Threshold for contrast limiting.  Higher values allow more
            contrast enhancement.  Must be strictly positive.
        tile_grid_size : list[int, int], default [8, 8]
            Number of tiles in the grid: ``[rows, cols]``.  Each dimension
            must be a positive integer.
    """

    FILTER_NAME: str = "clahe"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)

        self._clip_limit: float = float(self._params.get("clip_limit", 2.0))
        if self._clip_limit <= _MIN_CLIP_LIMIT:
            raise ValueError(
                f"CLAHEFilter: clip_limit must be > 0, got {self._clip_limit!r}"
            )

        raw_grid = self._params.get("tile_grid_size", [8, 8])
        try:
            self._tile_grid_size: tuple[int, int] = (int(raw_grid[0]), int(raw_grid[1]))
        except (TypeError, IndexError, ValueError) as exc:
            raise ValueError(
                "CLAHEFilter: tile_grid_size must be a two-element list [rows, cols]."
            ) from exc

        if any(v < _MIN_TILE_SIZE for v in self._tile_grid_size):
            raise ValueError(
                f"CLAHEFilter: tile_grid_size values must be >= 1, "
                f"got {self._tile_grid_size!r}"
            )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to the lightness channel of *image*.

        Parameters
        ----------
        image:
            Input RGB ``uint8`` array ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Contrast-enhanced RGB ``uint8`` array ``(H, W, 3)``.
        """
        clahe = cv2.createCLAHE(
            clipLimit=self._clip_limit,
            tileGridSize=self._tile_grid_size,
        )

        # Convert RGB → LAB, equalize L, convert back.
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_enhanced = clahe.apply(l_ch)
        lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
