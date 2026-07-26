"""Unit tests for ContrastFilter (filters.contrast).

The contrast transform is a simple affine operation so pixel-exact assertions
are possible for all test cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from comic_renderer.filters.contrast import ContrastFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid(v: int, h: int = 8, w: int = 8) -> np.ndarray:
    return np.full((h, w, 3), v, dtype=np.uint8)


def _pixel_image(r: int, g: int, b: int) -> np.ndarray:
    """Return a 1×1 image with the given pixel value."""
    return np.array([[[r, g, b]]], dtype=np.uint8)


def _contrast(alpha: float, beta: float, image: np.ndarray) -> np.ndarray:
    return ContrastFilter(params={"alpha": alpha, "beta": beta}).apply(image)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestContrastFilterContract:

    def test_output_is_uint8(self) -> None:
        assert _contrast(1.0, 0.0, _solid(100)).dtype == np.uint8

    def test_output_has_three_channels(self) -> None:
        result = _contrast(1.0, 0.0, _solid(100))
        assert result.ndim == 3 and result.shape[2] == 3

    def test_shape_preserved(self) -> None:
        image = np.zeros((24, 48, 3), dtype=np.uint8)
        assert _contrast(1.5, 10.0, image).shape == (24, 48, 3)

    def test_output_values_in_uint8_range(self) -> None:
        result = _contrast(3.0, 50.0, _solid(200))
        assert result.min() >= 0 and result.max() <= 255


# ---------------------------------------------------------------------------
# Pixel-exact arithmetic tests
# ---------------------------------------------------------------------------


class TestContrastFilterArithmetic:

    def test_identity_alpha_1_beta_0(self) -> None:
        """alpha=1, beta=0 → image is unchanged."""
        image = np.array([[[50, 100, 200]]], dtype=np.uint8)
        result = _contrast(1.0, 0.0, image)
        np.testing.assert_array_equal(result, image)

    def test_double_contrast(self) -> None:
        """alpha=2, beta=0: pixel 100 → 200."""
        result = _contrast(2.0, 0.0, _pixel_image(100, 50, 25))
        assert result[0, 0, 0] == 200
        assert result[0, 0, 1] == 100
        assert result[0, 0, 2] == 50

    def test_halve_contrast(self) -> None:
        """alpha=0.5, beta=0: pixel 200 → 100."""
        result = _contrast(0.5, 0.0, _pixel_image(200, 100, 0))
        assert result[0, 0, 0] == 100
        assert result[0, 0, 1] == 50
        assert result[0, 0, 2] == 0

    def test_positive_beta_brightens(self) -> None:
        """beta=50: pixel 100 → 150."""
        result = _contrast(1.0, 50.0, _solid(100))
        assert np.all(result == 150)

    def test_negative_beta_darkens(self) -> None:
        """beta=-50: pixel 100 → 50."""
        result = _contrast(1.0, -50.0, _solid(100))
        assert np.all(result == 50)

    def test_clipping_at_upper_bound(self) -> None:
        """alpha=3, beta=0: pixel 200 → clipped at 255, not 600."""
        result = _contrast(3.0, 0.0, _solid(200))
        assert np.all(result == 255)

    def test_clipping_at_lower_bound(self) -> None:
        """beta=-200: pixel 100 → clipped at 0, not -100."""
        result = _contrast(1.0, -200.0, _solid(100))
        assert np.all(result == 0)

    def test_alpha_zero_produces_black(self) -> None:
        """alpha=0: all pixels → beta (clipped to 0 when beta=0)."""
        result = _contrast(0.0, 0.0, _solid(255))
        assert np.all(result == 0)

    def test_alpha_zero_with_positive_beta(self) -> None:
        """alpha=0, beta=100: all pixels → 100."""
        result = _contrast(0.0, 100.0, _solid(255))
        assert np.all(result == 100)

    def test_does_not_modify_input_array(self) -> None:
        """apply() must not alter the input image in-place."""
        image = _solid(100)
        original_copy = image.copy()
        _contrast(2.0, 50.0, image)
        np.testing.assert_array_equal(image, original_copy)

    def test_channels_transformed_independently(self) -> None:
        """alpha=2, beta=0 should double all channels independently."""
        image = np.array([[[10, 50, 100]]], dtype=np.uint8)
        result = _contrast(2.0, 0.0, image)
        assert result[0, 0, 0] == 20
        assert result[0, 0, 1] == 100
        assert result[0, 0, 2] == 200


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestContrastFilterValidation:

    def test_negative_alpha_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            ContrastFilter(params={"alpha": -0.1})

    def test_default_alpha_and_beta(self) -> None:
        filt = ContrastFilter()
        assert filt._alpha == 1.2
        assert filt._beta == 0.0
