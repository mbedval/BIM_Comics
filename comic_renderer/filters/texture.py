"""TextureFilter – adds a procedural paper or grain texture to the image.

The texture is generated deterministically from a seed so that the same
seed always produces the same visual result regardless of image content.
This makes renders reproducible.

Texture modes
-------------
grain
    Fine, pixel-level noise blurred to the requested grain size.  Simulates
    film grain or sensor noise.

paper *(default)*
    Multi-scale noise combining a fine layer with an upscaled coarse layer.
    Produces an irregular paper-fibre feel at minimal computational cost.

Blend formula
-------------
The texture ``T`` is generated in the range [0, 255].  An offset centred on
zero is derived and scaled by ``strength``:

    offset = (T − 128) × strength
    result = clamp(image + offset, 0, 255)

At ``strength = 0`` the filter is a strict no-op.
At ``strength = 0.15`` (default) only a subtle texture is visible.
At ``strength = 0.5`` the texture is prominently visible.

JSON preset usage
-----------------
.. code-block:: json

    {
        "filter": "texture",
        "params": {"mode": "paper", "strength": 0.12, "seed": 42}
    }
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from comic_renderer.filters.base import BaseFilter

_VALID_MODES: frozenset[str] = frozenset({"grain", "paper"})


class TextureFilter(BaseFilter):
    """Overlay a procedural texture onto the image.

    Parameters
    ----------
    params:
        mode : str, default ``"paper"``
            Texture style: ``"paper"`` or ``"grain"``.
        strength : float, default 0.15
            Texture opacity in [0, 1].  0 = no effect.
        seed : int, default 42
            PRNG seed for reproducible renders.
        grain_size : int, default 2
            Blur radius for ``"grain"`` mode (ignored for ``"paper"``).
            Larger values produce coarser grain.
    """

    FILTER_NAME: str = "texture"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)

        self._mode: str = self._params.get("mode", "paper")
        if self._mode not in _VALID_MODES:
            raise ValueError(
                f"TextureFilter: unknown mode {self._mode!r}. "
                f"Valid options: {sorted(_VALID_MODES)}"
            )

        self._strength: float = float(self._params.get("strength", 0.15))
        if not (0.0 <= self._strength <= 1.0):
            raise ValueError(
                f"TextureFilter: strength must be in [0, 1], got {self._strength!r}"
            )

        self._seed: int = int(self._params.get("seed", 42))

        self._grain_size: int = int(self._params.get("grain_size", 2))
        if self._grain_size < 1:
            raise ValueError(
                f"TextureFilter: grain_size must be >= 1, got {self._grain_size!r}"
            )

    # ------------------------------------------------------------------
    # Public apply
    # ------------------------------------------------------------------

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Blend a procedural texture onto *image*.

        Parameters
        ----------
        image:
            Input RGB ``uint8`` array ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Textured RGB ``uint8`` array ``(H, W, 3)``.
        """
        if self._strength == 0.0:
            return image.copy()

        h, w = image.shape[:2]
        texture = self._generate(h, w)

        # Centre-offset blend: positive values brighten, negative darken.
        offset = (texture.astype(np.float64) - 128.0) * self._strength
        result = image.astype(np.float64) + offset[:, :, np.newaxis]
        return np.clip(result, 0.0, 255.0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Private texture generators
    # ------------------------------------------------------------------

    def _generate(self, height: int, width: int) -> np.ndarray:
        """Return a single-channel uint8 texture map of shape ``(H, W)``."""
        rng = np.random.default_rng(self._seed)

        if self._mode == "grain":
            return self._generate_grain(rng, height, width)
        return self._generate_paper(rng, height, width)

    def _generate_grain(
        self, rng: np.random.Generator, height: int, width: int
    ) -> np.ndarray:
        """Fine-grained noise, optionally smoothed."""
        noise = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
        if self._grain_size > 1:
            k = 2 * self._grain_size + 1  # Gaussian kernel size (must be odd)
            noise = cv2.GaussianBlur(noise, (k, k), float(self._grain_size) * 0.4)
        return noise

    def _generate_paper(
        self, rng: np.random.Generator, height: int, width: int
    ) -> np.ndarray:
        """Multi-scale paper fibre texture."""
        # Fine layer – slight variation around mid-grey.
        fine = rng.integers(110, 147, size=(height, width), dtype=np.uint8)

        # Coarse layer – upsampled from a small random array.
        coarse_h = max(height // 8, 2)
        coarse_w = max(width // 8, 2)
        coarse_small = rng.integers(96, 161, size=(coarse_h, coarse_w), dtype=np.uint8)
        coarse = cv2.resize(
            coarse_small, (width, height), interpolation=cv2.INTER_LINEAR
        )

        # Blend fine + coarse, then smooth.
        combined = cv2.addWeighted(fine, 0.55, coarse, 0.45, 0)
        return cv2.GaussianBlur(combined, (7, 7), 1.5)
