"""ContrastFilter – linear contrast and brightness adjustment.

Applies a pixel-wise affine transform to every channel:

    result[y, x, c] = clamp(α · image[y, x, c] + β, 0, 255)

where:

* **α (alpha)** – contrast multiplier.
  α = 1.0 → no change; α > 1.0 → higher contrast; 0 < α < 1.0 → lower.
* **β (beta)** – brightness offset (signed integer).
  β = 0 → no change; β > 0 → brighter; β < 0 → darker.

The computation is performed in float64 to avoid intermediate overflow,
then clamped to ``[0, 255]`` and cast back to uint8.

JSON preset usage
-----------------
.. code-block:: json

    {"filter": "contrast", "params": {"alpha": 1.3, "beta": 5}}
"""

from __future__ import annotations

from typing import Any

import numpy as np

from comic_renderer.filters.base import BaseFilter


class ContrastFilter(BaseFilter):
    """Adjust contrast and brightness with a linear transform.

    Parameters
    ----------
    params:
        alpha : float, default 1.2
            Contrast multiplier.  Must be >= 0.
        beta : float, default 0.0
            Brightness offset.  Negative values darken; positive brighten.
    """

    FILTER_NAME: str = "contrast"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)

        self._alpha: float = float(self._params.get("alpha", 1.2))
        if self._alpha < 0.0:
            raise ValueError(
                f"ContrastFilter: alpha must be >= 0, got {self._alpha!r}"
            )

        self._beta: float = float(self._params.get("beta", 0.0))

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply contrast and brightness scaling to *image*.

        Parameters
        ----------
        image:
            Input RGB ``uint8`` array ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Adjusted RGB ``uint8`` array ``(H, W, 3)``.
        """
        # Use float64 to avoid uint8 overflow before clipping.
        adjusted = image.astype(np.float64) * self._alpha + self._beta
        return np.clip(adjusted, 0.0, 255.0).astype(np.uint8)
