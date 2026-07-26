"""SharpenFilter – unsharp masking sharpening.

Unsharp masking is the standard technique for detail enhancement in both
darkroom photography and digital imaging:

1. Create a blurred version of the image with a Gaussian kernel.
2. Compute the *high-frequency residual* (detail layer):
       detail = image − blurred
3. Add the detail back with a strength multiplier:
       result = image + strength · detail
             = (1 + strength) · image − strength · blurred

Higher ``strength`` values produce crisper edges at the risk of ringing
artefacts.  Setting ``strength = 0`` leaves the image unchanged.

JSON preset usage
-----------------
.. code-block:: json

    {
        "filter": "sharpen",
        "params": {"strength": 1.0, "kernel_size": 5, "sigma": 1.0}
    }
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from comic_renderer.filters.base import BaseFilter


class SharpenFilter(BaseFilter):
    """Sharpen an image using unsharp masking.

    Parameters
    ----------
    params:
        strength : float, default 1.0
            How strongly to amplify high-frequency details.  Must be >= 0.
            0 = no change; 1.0 = standard sharpening; >2.0 = aggressive.
        kernel_size : int, default 5
            Size of the Gaussian blur kernel (must be an odd integer >= 3).
        sigma : float, default 1.0
            Standard deviation of the Gaussian kernel.  Must be > 0.
    """

    FILTER_NAME: str = "sharpen"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)

        self._strength: float = float(self._params.get("strength", 1.0))
        if self._strength < 0.0:
            raise ValueError(
                f"SharpenFilter: strength must be >= 0, got {self._strength!r}"
            )

        self._kernel_size: int = int(self._params.get("kernel_size", 5))
        if self._kernel_size < 3:
            raise ValueError(
                f"SharpenFilter: kernel_size must be >= 3, got {self._kernel_size!r}"
            )
        if self._kernel_size % 2 == 0:
            raise ValueError(
                f"SharpenFilter: kernel_size must be odd, got {self._kernel_size!r}"
            )

        self._sigma: float = float(self._params.get("sigma", 1.0))
        if self._sigma <= 0.0:
            raise ValueError(
                f"SharpenFilter: sigma must be > 0, got {self._sigma!r}"
            )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply unsharp masking to *image*.

        Parameters
        ----------
        image:
            Input RGB ``uint8`` array ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Sharpened RGB ``uint8`` array ``(H, W, 3)``.
        """
        blurred = cv2.GaussianBlur(
            image,
            (self._kernel_size, self._kernel_size),
            self._sigma,
        )
        # addWeighted: dst = src1 * alpha + src2 * beta + gamma
        # dst = image * (1 + strength) + blurred * (-strength)
        # Saturates to [0, 255] automatically and returns uint8.
        sharpened: np.ndarray = cv2.addWeighted(
            image,
            1.0 + self._strength,
            blurred,
            -self._strength,
            0.0,
        )
        return sharpened
