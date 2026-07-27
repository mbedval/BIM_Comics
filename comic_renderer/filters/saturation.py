"""SaturationFilter – boost or reduce colour saturation via HSV scaling.

Converts the image to HSV colour space, scales the S (saturation) channel by
``scale``, then converts back to RGB.  Values above 1.0 produce vivid,
punchy colours ideal for cartoon/meme styles; values below 1.0 produce muted,
desaturated tones.

Algorithm
---------
1. RGB → HSV (float32, H∈[0,179], S∈[0,255], V∈[0,255] via OpenCV).
2. S channel = clip(S × scale, 0, 255).
3. HSV → RGB.

JSON preset usage
-----------------
.. code-block:: json

    {"filter": "saturation", "params": {"scale": 1.6}}
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from comic_renderer.filters.base import BaseFilter

_MIN_SCALE: float = 0.0
_MAX_SCALE: float = 5.0


class SaturationFilter(BaseFilter):
    """Scale the HSV saturation channel by a constant factor.

    Parameters
    ----------
    params:
        scale : float, default 1.5
            Saturation multiplier.  Must be in [0.0, 5.0].
            1.0 = no change.  >1.0 = more vivid.  <1.0 = more muted.
    """

    FILTER_NAME: str = "saturation"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._scale: float = float(self._params.get("scale", 1.5))
        if not (_MIN_SCALE <= self._scale <= _MAX_SCALE):
            raise ValueError(
                f"SaturationFilter: scale must be in [{_MIN_SCALE}, {_MAX_SCALE}], "
                f"got {self._scale!r}"
            )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Boost or reduce saturation of *image*.

        Parameters
        ----------
        image:
            Input RGB ``uint8`` array ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            RGB ``uint8`` array with adjusted saturation.
        """
        # OpenCV expects BGR for cvtColor
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Scale the S channel and clip
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self._scale, 0.0, 255.0)

        hsv_u8 = hsv.astype(np.uint8)
        bgr_out = cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2BGR)
        return cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)
