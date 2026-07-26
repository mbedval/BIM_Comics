"""MorphologyFilter – morphological image processing operations.

Morphological operations work on the shape and structure of bright regions
(high-value pixels) within an image.  In comic rendering they are useful for:

* **dilate** – expands bright regions; thickens ink lines when applied to
  inverted images; creates a "swollen" look on posterized art.
* **erode** – shrinks bright regions; thins lines; removes small bright
  specks of noise.
* **close** *(default)* – dilation followed by erosion.  Fills small dark
  gaps in bright regions; smooths contours.  Very useful for sealing broken
  ink outlines.
* **open** – erosion followed by dilation.  Removes small bright noise
  blobs while leaving larger shapes intact.

Structuring element shapes
--------------------------
``rect``
    Solid rectangular kernel.  Square, hard-edged effect.

``ellipse`` *(default)*
    Circular/elliptical kernel.  Produces a rounder, more natural result.

``cross``
    Diamond-shaped plus sign.  Processes along cardinal directions only.

JSON preset usage
-----------------
.. code-block:: json

    {
        "filter": "morphology",
        "params": {
            "operation": "close",
            "kernel_size": 3,
            "iterations": 1,
            "kernel_shape": "ellipse"
        }
    }
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from comic_renderer.filters.base import BaseFilter

_VALID_OPERATIONS: frozenset[str] = frozenset({"dilate", "erode", "close", "open"})
_VALID_KERNEL_SHAPES: frozenset[str] = frozenset({"rect", "ellipse", "cross"})

_CV_MORPH_OP: dict[str, int] = {
    "dilate": cv2.MORPH_DILATE,
    "erode":  cv2.MORPH_ERODE,
    "close":  cv2.MORPH_CLOSE,
    "open":   cv2.MORPH_OPEN,
}

_CV_KERNEL_SHAPE: dict[str, int] = {
    "rect":    cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross":   cv2.MORPH_CROSS,
}


class MorphologyFilter(BaseFilter):
    """Apply a morphological operation to the image.

    Parameters
    ----------
    params:
        operation : str, default ``"close"``
            Morphological operation: ``"dilate"``, ``"erode"``,
            ``"close"``, or ``"open"``.
        kernel_size : int, default 3
            Side length of the structuring element (must be >= 1).
        iterations : int, default 1
            Number of times to repeat the operation.  Must be >= 1.
        kernel_shape : str, default ``"ellipse"``
            Shape of the structuring element: ``"rect"``, ``"ellipse"``,
            or ``"cross"``.
    """

    FILTER_NAME: str = "morphology"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)

        self._operation: str = self._params.get("operation", "close")
        if self._operation not in _VALID_OPERATIONS:
            raise ValueError(
                f"MorphologyFilter: unknown operation {self._operation!r}. "
                f"Valid options: {sorted(_VALID_OPERATIONS)}"
            )

        self._kernel_size: int = int(self._params.get("kernel_size", 3))
        if self._kernel_size < 1:
            raise ValueError(
                f"MorphologyFilter: kernel_size must be >= 1, "
                f"got {self._kernel_size!r}."
            )

        self._iterations: int = int(self._params.get("iterations", 1))
        if self._iterations < 1:
            raise ValueError(
                f"MorphologyFilter: iterations must be >= 1, "
                f"got {self._iterations!r}."
            )

        self._kernel_shape: str = self._params.get("kernel_shape", "ellipse")
        if self._kernel_shape not in _VALID_KERNEL_SHAPES:
            raise ValueError(
                f"MorphologyFilter: unknown kernel_shape {self._kernel_shape!r}. "
                f"Valid options: {sorted(_VALID_KERNEL_SHAPES)}"
            )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the morphological operation to *image*.

        Parameters
        ----------
        image:
            Input RGB ``uint8`` array ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Morphologically-processed RGB ``uint8`` array ``(H, W, 3)``.
        """
        kernel = cv2.getStructuringElement(
            _CV_KERNEL_SHAPE[self._kernel_shape],
            (self._kernel_size, self._kernel_size),
        )
        return cv2.morphologyEx(
            image,
            _CV_MORPH_OP[self._operation],
            kernel,
            iterations=self._iterations,
        )
