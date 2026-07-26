"""Unit tests for CLAHEFilter (filters.clahe)."""

from __future__ import annotations

import numpy as np
import pytest

from comic_renderer.filters.clahe import CLAHEFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gradient_image(h: int = 32, w: int = 32) -> np.ndarray:
    """Return an RGB image with a horizontal grayscale gradient."""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    gray = np.tile(row, (h, 1))
    return np.stack([gray, gray, gray], axis=2)


def _uniform_image(value: int = 128, h: int = 16, w: int = 16) -> np.ndarray:
    """Return a solid-colour image."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _rgb_image(h: int = 32, w: int = 32) -> np.ndarray:
    """Return a non-grayscale RGB image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 200, w, dtype=np.uint8)    # R gradient
    img[:, :, 1] = np.linspace(50, 255, w, dtype=np.uint8)   # G gradient
    img[:, :, 2] = 100                                         # B constant
    return img


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestCLAHEFilterContract:

    def test_output_is_uint8(self) -> None:
        result = CLAHEFilter().apply(_gradient_image())
        assert result.dtype == np.uint8

    def test_output_has_three_channels(self) -> None:
        result = CLAHEFilter().apply(_gradient_image())
        assert result.ndim == 3 and result.shape[2] == 3

    def test_shape_preserved(self) -> None:
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        result = CLAHEFilter().apply(image)
        assert result.shape == (48, 64, 3)

    def test_output_values_in_uint8_range(self) -> None:
        result = CLAHEFilter().apply(_gradient_image())
        assert result.min() >= 0
        assert result.max() <= 255


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestCLAHEFilterFunctional:

    def test_does_not_crash_on_uniform_image(self) -> None:
        """A uniform image should not raise even if histogram is trivial."""
        result = CLAHEFilter().apply(_uniform_image(128))
        assert result.dtype == np.uint8

    def test_gradient_image_output_differs_from_input(self) -> None:
        """CLAHE must change at least some pixel values on a non-trivial input."""
        image = _gradient_image()
        result = CLAHEFilter().apply(image)
        # The output should not be identical to the input.
        assert not np.array_equal(result, image)

    def test_rgb_image_output_is_valid(self) -> None:
        """CLAHE on a colour image must return a valid uint8 3-channel array."""
        result = CLAHEFilter().apply(_rgb_image())
        assert result.dtype == np.uint8
        assert result.shape[2] == 3

    def test_grayscale_input_stays_near_grayscale(self) -> None:
        """For a grayscale input (R=G=B), R and G output channels are close."""
        gray = _gradient_image()
        result = CLAHEFilter().apply(gray)
        diff = np.abs(result[:, :, 0].astype(int) - result[:, :, 1].astype(int))
        # Due to LAB round-trip, channels may differ by a few values.
        assert diff.max() <= 5

    def test_clip_limit_affects_output(self) -> None:
        """Different clip limits must produce different outputs on a realistic image.

        A simple linear gradient is already maximally spread and CLAHE may
        produce the same result regardless of clip_limit.  Use a bimodal
        distribution (mostly dark, small bright patch) instead.
        """
        # Build a bimodal image: mostly 20, with a 4x4 bright patch at 240.
        image = np.full((32, 32, 3), 20, dtype=np.uint8)
        image[14:18, 14:18, :] = 240  # small bright region

        result_low = CLAHEFilter(params={"clip_limit": 0.5}).apply(image)
        result_high = CLAHEFilter(params={"clip_limit": 40.0}).apply(image)
        assert not np.array_equal(result_low, result_high)

    def test_tile_grid_size_accepted(self) -> None:
        """Non-default tile grid sizes must not raise."""
        result = CLAHEFilter(params={"tile_grid_size": [4, 4]}).apply(_gradient_image())
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestCLAHEFilterValidation:

    def test_negative_clip_limit_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="clip_limit"):
            CLAHEFilter(params={"clip_limit": -1.0})

    def test_zero_clip_limit_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="clip_limit"):
            CLAHEFilter(params={"clip_limit": 0.0})

    def test_invalid_tile_grid_size_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="tile_grid_size"):
            CLAHEFilter(params={"tile_grid_size": [0, 8]})

    def test_non_list_tile_grid_size_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="tile_grid_size"):
            CLAHEFilter(params={"tile_grid_size": "8x8"})

    def test_default_params_do_not_raise(self) -> None:
        """Default construction must succeed."""
        filt = CLAHEFilter()
        assert filt is not None
