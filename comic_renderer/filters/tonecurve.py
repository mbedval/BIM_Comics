"""ToneCurveFilter – adjusts image tones using a piece-wise linear lookup table.

Allows custom mapping of input pixel values to output pixel values. The
user defines a set of control points, which are interpolated to construct
a 256-value lookup table (LUT).

JSON preset usage
-----------------
.. code-block:: json

    {
        "filter": "tonecurve",
        "params": {
            "points": [
                [0, 0],
                [64, 40],
                [128, 128],
                [192, 215],
                [255, 255]
            ]
        }
    }
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from comic_renderer.filters.base import BaseFilter


class ToneCurveFilter(BaseFilter):
    """Adjust contrast, brightness, or gamma mapping via lookup tables (LUTs).

    Parameters
    ----------
    params:
        points : list[list[int, int]], default [[0, 0], [255, 255]]
            List of [x, y] control points. Must contain at least two points.
            Values must be in [0, 255]. The points will be sorted by x.
            The first point's x must be 0, and the last point's x must be 255.
            All x values must be strictly increasing.
    """

    FILTER_NAME: str = "tonecurve"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)

        raw_points = self._params.get("points", [[0, 0], [255, 255]])
        if not isinstance(raw_points, list) or len(raw_points) < 2:
            raise ValueError(
                "ToneCurveFilter: points must be a list containing at least 2 points."
            )

        # Parse and sort points
        parsed = []
        for p in raw_points:
            try:
                x, y = int(p[0]), int(p[1])
            except (TypeError, IndexError, ValueError) as exc:
                raise ValueError(
                    f"ToneCurveFilter: invalid point format {p!r}."
                ) from exc

            if not (0 <= x <= 255) or not (0 <= y <= 255):
                raise ValueError(
                    f"ToneCurveFilter: coordinates must be in [0, 255], got ({x}, {y})."
                )
            parsed.append((x, y))

        # Sort points by x-coordinate
        parsed.sort(key=lambda pt: pt[0])

        # Validate bounds and monotonicity
        if parsed[0][0] != 0:
            raise ValueError(
                f"ToneCurveFilter: the first point must start at x=0, got x={parsed[0][0]}."
            )
        if parsed[-1][0] != 255:
            raise ValueError(
                f"ToneCurveFilter: the last point must end at x=255, got x={parsed[-1][0]}."
            )

        for i in range(len(parsed) - 1):
            if parsed[i][0] == parsed[i + 1][0]:
                raise ValueError(
                    f"ToneCurveFilter: duplicate x-coordinate found at x={parsed[i][0]}."
                )

        self._points = parsed
        self._lut = self._build_lut()

    def _build_lut(self) -> np.ndarray:
        """Interpolate the points to create a 256-byte lookup table."""
        xs = [pt[0] for pt in self._points]
        ys = [pt[1] for pt in self._points]

        # Interpolate output values for all inputs from 0 to 255
        lut_vals = np.interp(np.arange(256), xs, ys)
        
        # Round and clip to uint8 bounds
        return np.clip(np.round(lut_vals), 0, 255).astype(np.uint8)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the tone curve lookup table to *image*.

        Parameters
        ----------
        image:
            Input RGB ``uint8`` array ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Tone-mapped RGB ``uint8`` array ``(H, W, 3)``.
        """
        # cv2.LUT expects the lookup table to have shape (256, 1) or (1, 256, 1)
        # or matching channel shapes. Shape (256, 1) will apply the same LUT
        # to all channels.
        lut_reshaped = self._lut[:, np.newaxis]
        return cv2.LUT(image, lut_reshaped)
