"""Pass-through filter – returns an unchanged copy of the input image.

Purpose
-------
This filter serves two roles:

1. **Testing** – a concrete ``BaseFilter`` implementation that is always
   available for unit and integration tests without depending on any
   real image-processing logic.

2. **Preset stubbing** – all preset JSONs reference this filter as a
   placeholder until the real filters are implemented in Milestones 3–4.
   Switching a preset to real filters requires only a JSON edit; no Python
   changes are needed.

Usage in a preset JSON::

    {"filter": "passthrough", "params": {}}
"""

from __future__ import annotations

from typing import Any

import numpy as np

from comic_renderer.filters.base import BaseFilter


class PassThroughFilter(BaseFilter):
    """Returns a copy of the input image with no modifications.

    Parameters
    ----------
    params:
        Accepted for interface conformance; all keys are ignored.
    """

    FILTER_NAME: str = "passthrough"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Return a copy of *image* unmodified.

        Parameters
        ----------
        image:
            Input RGB ``uint8`` array.

        Returns
        -------
        np.ndarray
            Identical copy of *image*.
        """
        return image.copy()
