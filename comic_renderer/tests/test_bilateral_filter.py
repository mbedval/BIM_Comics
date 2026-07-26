"""Unit tests for BilateralFilter (filters.bilateral)."""

from __future__ import annotations

import numpy as np
import pytest

from comic_renderer.filters.bilateral import BilateralFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid(value: int = 128, h: int = 16, w: int = 16) -> np.ndarray:
    """Return a solid uniform RGB image."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _noisy_image(h: int = 32, w: int = 32) -> np.ndarray:
    """Return a gray image with some random uniform noise."""
    rng = np.random.default_rng(42)
    gray = rng.integers(100, 150, size=(h, w), dtype=np.uint8)
    return np.stack([gray, gray, gray], axis=2)


def _apply(image: np.ndarray, **params) -> np.ndarray:
    return BilateralFilter(params=params).apply(image)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestBilateralFilterContract:

    def test_output_is_uint8(self) -> None:
        assert _apply(_noisy_image()).dtype == np.uint8

    def test_output_has_three_channels(self) -> None:
        result = _apply(_noisy_image())
        assert result.ndim == 3 and result.shape[2] == 3

    def test_shape_preserved(self) -> None:
        image = np.zeros((24, 48, 3), dtype=np.uint8)
        result = _apply(image)
        assert result.shape == (24, 48, 3)

    def test_output_values_in_uint8_range(self) -> None:
        result = _apply(_noisy_image())
        assert result.min() >= 0 and result.max() <= 255


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestBilateralFilterFunctional:

    def test_uniform_image_mostly_unchanged(self) -> None:
        """A uniform image should stay uniform after bilateral filtering."""
        image = _solid(128)
        result = _apply(image, diameter=5, sigma_color=50, sigma_space=50)
        # Allow tiny difference due to float calculation in OpenCV
        diff = np.abs(result.astype(int) - image.astype(int))
        assert diff.max() <= 1

    def test_bilateral_smoothes_noisy_image(self) -> None:
        """Bilateral filter must reduce variance in a noisy image (smoothing effect)."""
        image = _noisy_image()
        result = _apply(image, diameter=9, sigma_color=80, sigma_space=80)
        
        orig_std = image.std()
        result_std = result.std()
        
        # Output must be smoother (have lower standard deviation) than input
        assert result_std < orig_std

    def test_multiple_iterations_run_successfully(self) -> None:
        image = _noisy_image()
        result1 = _apply(image, diameter=5, sigma_color=30, sigma_space=30, iterations=1)
        result3 = _apply(image, diameter=5, sigma_color=30, sigma_space=30, iterations=3)
        assert result1.dtype == np.uint8
        assert result3.dtype == np.uint8
        # Smoothness should generally increase or change across iterations
        assert not np.array_equal(result1, result3)

    def test_returns_copy(self) -> None:
        image = _noisy_image()
        result = _apply(image)
        assert result is not image


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestBilateralFilterValidation:

    def test_zero_sigma_color_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="sigma_color"):
            BilateralFilter(params={"sigma_color": 0.0})

    def test_negative_sigma_color_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="sigma_color"):
            BilateralFilter(params={"sigma_color": -1.5})

    def test_zero_sigma_space_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="sigma_space"):
            BilateralFilter(params={"sigma_space": 0.0})

    def test_negative_sigma_space_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="sigma_space"):
            BilateralFilter(params={"sigma_space": -10.0})

    def test_zero_iterations_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="iterations"):
            BilateralFilter(params={"iterations": 0})

    def test_negative_iterations_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="iterations"):
            BilateralFilter(params={"iterations": -2})

    def test_default_params_succeed(self) -> None:
        filt = BilateralFilter()
        assert filt._diameter == 9
        assert filt._sigma_color == 75.0
        assert filt._sigma_space == 75.0
        assert filt._iterations == 1
