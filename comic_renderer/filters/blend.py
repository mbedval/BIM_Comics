"""BlendFilter – self-blend tone-mapping modes.

In a sequential single-input pipeline a blend filter cannot receive two
separate image sources.  Instead, BlendFilter applies a *self-blend* – it
mixes the image with a version of itself transformed by a Photoshop-style
blend mode.  This produces the tonal effects that are valuable in
comic-style rendering:

Modes
-----
overlay *(default)*
    Multiply for dark tones, Screen for light tones.  Strong S-curve
    contrast boost.  The most dramatic effect.

soft_light
    Gentler S-curve (Pegtop formula).  Suitable when overlay is too harsh.

multiply
    Squares the channel values (normalised).  Darkens and deepens shadows.
    Useful for noir and dark graphic novel styles.

screen
    Inverse of multiply.  Lightens and lifts highlights.

Strength
--------
``strength`` controls the interpolation weight between the original image
and the blended result::

    output = (1 − strength) × original + strength × blend(original)

At ``strength = 0`` the output is identical to the input.
At ``strength = 1`` the output is purely the blend result.

JSON preset usage
-----------------
.. code-block:: json

    {"filter": "blend", "params": {"mode": "overlay", "strength": 0.35}}
"""

from __future__ import annotations

from typing import Any

import numpy as np

from comic_renderer.filters.base import BaseFilter

_VALID_MODES: frozenset[str] = frozenset({"overlay", "soft_light", "multiply", "screen"})


class BlendFilter(BaseFilter):
    """Apply a self-blend tone-mapping to the image.

    Parameters
    ----------
    params:
        mode : str, default ``"overlay"``
            Blend algorithm: ``"overlay"``, ``"soft_light"``, ``"multiply"``,
            or ``"screen"``.
        strength : float, default 0.5
            Interpolation weight in [0, 1].  0 = no effect; 1 = full blend.
    """

    FILTER_NAME: str = "blend"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)

        self._mode: str = self._params.get("mode", "overlay")
        if self._mode not in _VALID_MODES:
            raise ValueError(
                f"BlendFilter: unknown mode {self._mode!r}. "
                f"Valid options: {sorted(_VALID_MODES)}"
            )

        self._strength: float = float(self._params.get("strength", 0.5))
        if not (0.0 <= self._strength <= 1.0):
            raise ValueError(
                f"BlendFilter: strength must be in [0, 1], got {self._strength!r}"
            )

    # ------------------------------------------------------------------
    # Public apply
    # ------------------------------------------------------------------

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the self-blend to *image*.

        Parameters
        ----------
        image:
            Input RGB ``uint8`` array ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Tone-mapped RGB ``uint8`` array ``(H, W, 3)``.
        """
        if self._strength == 0.0:
            return image.copy()

        img_f = image.astype(np.float64)

        blended = self._apply_mode(img_f)

        # Linear interpolation between original and blended result.
        result = img_f * (1.0 - self._strength) + blended * self._strength
        return np.clip(result, 0.0, 255.0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Blend mode implementations (all vectorised, work in [0, 255] floats)
    # ------------------------------------------------------------------

    def _apply_mode(self, img: np.ndarray) -> np.ndarray:
        if self._mode == "overlay":
            return self._overlay(img, img)
        if self._mode == "soft_light":
            return self._soft_light(img, img)
        if self._mode == "multiply":
            return self._multiply(img, img)
        # screen
        return self._screen(img, img)

    @staticmethod
    def _overlay(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Multiply below 128, Screen above 128."""
        result = np.where(
            base < 128.0,
            2.0 * base * blend / 255.0,
            255.0 - 2.0 * (255.0 - base) * (255.0 - blend) / 255.0,
        )
        return result

    @staticmethod
    def _soft_light(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Pegtop soft-light formula (smooth S-curve)."""
        b = blend / 255.0
        c = base / 255.0
        result_n = (1.0 - 2.0 * b) * c**2 + 2.0 * b * c
        return result_n * 255.0

    @staticmethod
    def _multiply(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Darken – analogous to pigment mixing."""
        return base * blend / 255.0

    @staticmethod
    def _screen(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Lighten – inverse of multiply."""
        return 255.0 - (255.0 - base) * (255.0 - blend) / 255.0
