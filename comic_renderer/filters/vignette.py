"""VignetteFilter – adds a dark or colored vignette shading effect around image borders.

Simulates the light fall-off towards the edges of camera lenses. It creates
a radial mask centered on the image canvas, fading out towards the edges.

JSON preset usage
-----------------
.. code-block:: json

    {
        "filter": "vignette",
        "params": {
            "strength": 0.5,
            "radius": 1.0,
            "color": [0, 0, 0]
        }
    }
"""

from __future__ import annotations

from typing import Any

import numpy as np

from comic_renderer.filters.base import BaseFilter


class VignetteFilter(BaseFilter):
    """Apply a radial vignette shading effect.

    Parameters
    ----------
    params:
        strength : float, default 0.5
            Intensity/opacity of the vignette in [0, 1]. 0 = no effect.
        radius : float, default 1.0
            Relative radius of the vignette spread in [0.1, 3.0].
            Higher values push the vignette further out.
        color : list[int, int, int], default [0, 0, 0]
            RGB tuple/list of the vignette boundary color. Values in [0, 255].
    """

    FILTER_NAME: str = "vignette"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)

        self._strength: float = float(self._params.get("strength", 0.5))
        if not (0.0 <= self._strength <= 1.0):
            raise ValueError(
                f"VignetteFilter: strength must be in [0, 1], got {self._strength!r}"
            )

        self._radius: float = float(self._params.get("radius", 1.0))
        if self._radius < 0.1:
            raise ValueError(
                f"VignetteFilter: radius must be >= 0.1, got {self._radius!r}"
            )

        raw_color = self._params.get("color", [0, 0, 0])
        try:
            self._color: np.ndarray = np.array(
                [int(raw_color[0]), int(raw_color[1]), int(raw_color[2])],
                dtype=np.float64,
            )
        except (TypeError, IndexError, ValueError) as exc:
            raise ValueError(
                "VignetteFilter: color must be a three-element list [R, G, B]."
            ) from exc

        if np.any(self._color < 0) or np.any(self._color > 255):
            raise ValueError(
                f"VignetteFilter: color components must be in [0, 255], got {raw_color!r}."
            )

        # Cache variables for shape-based performance optimizations
        self._cached_shape: tuple[int, int] | None = None
        self._cached_mask: np.ndarray | None = None

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the radial vignette to *image*.

        Parameters
        ----------
        image:
            Input RGB ``uint8`` array ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Vignetted RGB ``uint8`` array ``(H, W, 3)``.
        """
        if self._strength == 0.0:
            return image.copy()

        h, w = image.shape[:2]

        if self._cached_shape != (h, w) or self._cached_mask is None:
            self._cached_shape = (h, w)
            # Generate coordinate indices
            y_indices, x_indices = np.indices((h, w), dtype=np.float64)

            # Center coordinates
            cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

            # Normalized coordinates: center is (0, 0), corners can be up to ~1.41
            dy = (y_indices - cy) / (h / 2.0)
            dx = (x_indices - cx) / (w / 2.0)

            # Radial distance from the center
            d = np.sqrt(dx**2 + dy**2)

            # Transmission factor t: 1.0 at center, decays quadratically outward
            t = 1.0 - (d / self._radius) ** 2
            t = np.clip(t, 0.0, 1.0)

            # Vignette mask factor v: 1.0 at center, (1 - strength) at outer boundary
            v = 1.0 - (1.0 - t) * self._strength

            # Reshape v to support broadcasting across channels
            self._cached_mask = v[:, :, np.newaxis]

        # Blend: output = image * mask + color * (1 - mask)
        result = image.astype(np.float64) * self._cached_mask + self._color * (1.0 - self._cached_mask)
        return np.clip(result, 0.0, 255.0).astype(np.uint8)
