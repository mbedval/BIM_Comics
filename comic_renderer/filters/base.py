"""Abstract base class that every filter in the pipeline must implement.

Design contract
---------------
* Each concrete filter is a **single-responsibility** transformation.
* Filters receive an RGB ``uint8`` NumPy array and must return one.
* Filters never call other filters.
* Parameters come from the JSON preset and are stored in ``self._params``
  at construction time.  Validation of those parameters belongs in the
  concrete subclass ``__init__``.

Class-level constant
--------------------
Every concrete subclass MUST define a class-level string constant::

    FILTER_NAME: str = "my_filter"

This name is the registry key used in the JSON preset (e.g.
``{"filter": "my_filter", "params": {...}}``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseFilter(ABC):
    """Abstract base for all image filters.

    Parameters
    ----------
    params:
        Dictionary of keyword parameters sourced from the JSON preset step.
        Concrete subclasses may override ``__init__`` to unpack and validate
        these, calling ``super().__init__(params)`` first.
    """

    #: Subclasses MUST override this with a unique lowercase snake_case name.
    FILTER_NAME: str = ""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params: dict[str, Any] = params or {}

    @property
    def params(self) -> dict[str, Any]:
        """Read-only view of the parameters supplied to this filter instance."""
        return dict(self._params)

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the filter to *image* and return the result.

        Parameters
        ----------
        image:
            Input RGB ``uint8`` array with shape ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Output RGB ``uint8`` array.  Shape and dtype must be preserved
            unless the filter intentionally changes the channel count (e.g.
            grayscale → replicate to 3 channels).

        Notes
        -----
        Implementations should **not** modify *image* in-place.  Either
        return ``image.copy()`` (pass-through) or build a new array.
        """

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(params={self._params!r})"
