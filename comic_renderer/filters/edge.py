"""EdgeFilter – detects edges and burns them as dark ink lines onto the image.

Algorithm
---------
1. Convert the RGB input to grayscale for edge detection.
2. Optionally pre-blur to suppress sensor noise before detection.
3. Detect edges with one of three classical algorithms:

   * **canny** *(default)* – Canny two-threshold detector.  Produces a clean
     binary edge map.  Preferred for most comic styles.
   * **sobel** – Gradient magnitude from horizontal + vertical Sobel kernels.
     Produces softer, continuous-strength edges.
   * **laplacian** – Second-derivative Laplacian operator.  Detects fine
     detail; works best on low-noise, already-sharpened images.

4. Convert the edge map to an *ink factor*:
       ``ink_factor = 1 − (edge_map / 255) × blend_strength``
   Pixels with strong edges become darker; blend_strength=0 leaves the
   image unchanged.

5. Multiply every RGB channel by the ink factor, clip to [0, 255], cast to
   uint8.

JSON preset usage
-----------------
.. code-block:: json

    {
        "filter": "edge",
        "params": {
            "method": "canny",
            "low_threshold": 30,
            "high_threshold": 100,
            "blend_strength": 0.9,
            "blur_radius": 1
        }
    }
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from comic_renderer.filters.base import BaseFilter

_VALID_METHODS: frozenset[str] = frozenset({"canny", "sobel", "laplacian"})


class EdgeFilter(BaseFilter):
    """Detect edges and apply them as dark ink lines.

    Parameters
    ----------
    params:
        method : str, default ``"canny"``
            Edge detection algorithm: ``"canny"``, ``"sobel"``, or
            ``"laplacian"``.
        low_threshold : int, default 30
            Lower hysteresis threshold for Canny (ignored for Sobel/Laplacian).
        high_threshold : int, default 100
            Upper hysteresis threshold for Canny.  Must be > ``low_threshold``.
        ksize : int, default 3
            Sobel/Laplacian kernel size (odd integer, 1–7).  Ignored for Canny.
        blend_strength : float, default 1.0
            Edge line opacity in [0, 1].  0 = no edges; 1 = fully black edges.
        blur_radius : int, default 0
            Pre-blur radius applied before edge detection to reduce noise.
            0 = no blur; 1 = 3×3 Gaussian; 2 = 5×5 Gaussian; etc.
    """

    FILTER_NAME: str = "edge"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)

        self._method: str = self._params.get("method", "canny")
        if self._method not in _VALID_METHODS:
            raise ValueError(
                f"EdgeFilter: unknown method {self._method!r}. "
                f"Valid options: {sorted(_VALID_METHODS)}"
            )

        self._low_threshold: int = int(self._params.get("low_threshold", 30))
        self._high_threshold: int = int(self._params.get("high_threshold", 100))
        if self._low_threshold < 0 or self._high_threshold < 0:
            raise ValueError("EdgeFilter: thresholds must be non-negative.")
        if self._low_threshold >= self._high_threshold:
            raise ValueError(
                f"EdgeFilter: low_threshold ({self._low_threshold}) must be "
                f"< high_threshold ({self._high_threshold})."
            )

        self._ksize: int = int(self._params.get("ksize", 3))
        if self._ksize < 1 or self._ksize % 2 == 0:
            raise ValueError(
                f"EdgeFilter: ksize must be a positive odd integer, "
                f"got {self._ksize!r}."
            )

        self._blend_strength: float = float(self._params.get("blend_strength", 1.0))
        if not (0.0 <= self._blend_strength <= 1.0):
            raise ValueError(
                f"EdgeFilter: blend_strength must be in [0, 1], "
                f"got {self._blend_strength!r}."
            )

        self._blur_radius: int = int(self._params.get("blur_radius", 0))
        if self._blur_radius < 0:
            raise ValueError(
                f"EdgeFilter: blur_radius must be >= 0, got {self._blur_radius!r}."
            )

    # ------------------------------------------------------------------
    # Public apply
    # ------------------------------------------------------------------

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply edge detection and burn ink lines into *image*.

        Parameters
        ----------
        image:
            Input RGB ``uint8`` array ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            RGB ``uint8`` array with dark edge lines applied.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if self._blur_radius > 0:
            k = 2 * self._blur_radius + 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        edge_map = self._detect_edges(gray)

        # ink_factor: 1.0 where no edge, (1 − blend_strength) where strong edge.
        ink_factor = 1.0 - (edge_map.astype(np.float64) / 255.0) * self._blend_strength

        result = image.astype(np.float64) * ink_factor[:, :, np.newaxis]
        return np.clip(result, 0.0, 255.0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Private edge detection strategies
    # ------------------------------------------------------------------

    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """Return a single-channel uint8 edge map (bright = edge)."""
        if self._method == "canny":
            return cv2.Canny(gray, self._low_threshold, self._high_threshold)

        if self._method == "sobel":
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self._ksize)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self._ksize)
            magnitude = np.sqrt(sx**2 + sy**2)
        else:  # laplacian
            lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=self._ksize)
            magnitude = np.abs(lap)

        return cv2.normalize(
            magnitude, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
