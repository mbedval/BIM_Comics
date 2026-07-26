"""BilateralFilter – edge-preserving smoothing filter.

Replaces the intensity of each pixel with a weighted average of intensity
values from nearby pixels, where the weights depend on both coordinate
distance and intensity differences. This smooths out flat regions (reducing
noise/details) while keeping edges sharp, making it highly effective for
achieving cartoon and comic aesthetics.

JSON preset usage
-----------------
.. code-block:: json

    {
        "filter": "bilateral",
        "params": {
            "diameter": 9,
            "sigma_color": 75.0,
            "sigma_space": 75.0,
            "iterations": 1
        }
    }
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from comic_renderer.filters.base import BaseFilter


class BilateralFilter(BaseFilter):
    """Apply bilateral filtering to smooth textures while preserving edges.

    Parameters
    ----------
    params:
        diameter : int, default 9
            Diameter of each pixel neighborhood. If non-positive, it is
            computed automatically from sigma_space.
        sigma_color : float, default 75.0
            Filter sigma in the color space. A larger value means that
            farther colors within the pixel neighborhood will be mixed together.
            Must be > 0.
        sigma_space : float, default 75.0
            Filter sigma in the coordinate space. A larger value means that
            farther pixels will influence each other as long as their colors
            are close enough. Must be > 0.
        iterations : int, default 1
            Number of times to apply the filter sequentially. Must be >= 1.
    """

    FILTER_NAME: str = "bilateral"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)

        self._diameter: int = int(self._params.get("diameter", 9))

        self._sigma_color: float = float(self._params.get("sigma_color", 75.0))
        if self._sigma_color <= 0.0:
            raise ValueError(
                f"BilateralFilter: sigma_color must be > 0, got {self._sigma_color!r}"
            )

        self._sigma_space: float = float(self._params.get("sigma_space", 75.0))
        if self._sigma_space <= 0.0:
            raise ValueError(
                f"BilateralFilter: sigma_space must be > 0, got {self._sigma_space!r}"
            )

        self._iterations: int = int(self._params.get("iterations", 1))
        if self._iterations < 1:
            raise ValueError(
                f"BilateralFilter: iterations must be >= 1, got {self._iterations!r}"
            )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filtering to *image*.

        Parameters
        ----------
        image:
            Input RGB ``uint8`` array ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Smoothed RGB ``uint8`` array ``(H, W, 3)``.
        """
        current = image
        for _ in range(self._iterations):
            # cv2.bilateralFilter handles 8-bit 3-channel (RGB) images natively.
            current = cv2.bilateralFilter(
                current,
                self._diameter,
                self._sigma_color,
                self._sigma_space,
            )
        return current
