"""Unit tests for HalftoneFilter (filters.halftone)."""

from __future__ import annotations

import numpy as np
import pytest

from comic_renderer.filters.halftone import HalftoneFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid(value: int = 128, h: int = 32, w: int = 32) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _apply(image: np.ndarray, **params) -> np.ndarray:
    return HalftoneFilter(params=params).apply(image)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestHalftoneFilterContract:

    def test_output_is_uint8(self) -> None:
        assert _apply(_solid()).dtype == np.uint8

    def test_output_has_three_channels(self) -> None:
        result = _apply(_solid())
        assert result.ndim == 3 and result.shape[2] == 3

    def test_shape_preserved(self) -> None:
        image = np.zeros((24, 48, 3), dtype=np.uint8)
        result = _apply(image)
        assert result.shape == (24, 48, 3)

    def test_output_values_in_uint8_range(self) -> None:
        result = _apply(_solid(255), blend_strength=1.0)
        assert result.min() >= 0 and result.max() <= 255


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestHalftoneFilterFunctional:

    def test_blend_strength_zero_is_noop(self) -> None:
        """blend_strength=0 must return a copy identical to the input."""
        image = _solid(128)
        result = _apply(image, blend_strength=0.0)
        np.testing.assert_array_equal(result, image)

    def test_pure_white_image_produces_fully_white_halftone(self) -> None:
        """A pure white image (255) is always brighter than any threshold [0, 255),
        so it must yield pure white output at blend_strength=1."""
        image = _solid(255)
        result = _apply(image, blend_strength=1.0, size=8, angle=45)
        np.testing.assert_array_equal(result, 255)

    def test_pure_black_image_produces_fully_black_halftone(self) -> None:
        """A pure black image (0) is never brighter than the threshold (which is >= 0),
        so it must yield pure black output at blend_strength=1.
        Wait, at the cell corners, d = d_max, so threshold is 0. If gray = 0,
        0 > 0 is False, so it is still 0 (black). Thus all pixels should be 0."""
        image = _solid(0)
        result = _apply(image, blend_strength=1.0, size=8, angle=45)
        np.testing.assert_array_equal(result, 0)

    def test_halftone_pattern_contains_black_and_white(self) -> None:
        """A mid-grey image halftoned with blend_strength=1 must produce a binary
        output containing only 0 and 255."""
        image = _solid(128)
        result = _apply(image, blend_strength=1.0, size=8)
        
        # Verify that output values are only 0 or 255
        unique_vals = np.unique(result)
        for val in unique_vals:
            assert val in (0, 255)
        
        # Output should contain both 0 and 255
        assert 0 in unique_vals
        assert 255 in unique_vals

    def test_returns_copy(self) -> None:
        image = _solid(128)
        result = _apply(image)
        assert result is not image

    def test_cache_invalidation_on_shape_change(self) -> None:
        """Applying the same filter instance to different shapes must invalidate cache and work correctly."""
        filt = HalftoneFilter(params={"size": 4, "angle": 45, "blend_strength": 1.0})
        img1 = _solid(128, h=16, w=16)
        img2 = _solid(128, h=32, w=32)

        res1 = filt.apply(img1)
        res2 = filt.apply(img2)

        assert res1.shape == (16, 16, 3)
        assert res2.shape == (32, 32, 3)
        
        # Verify both contain halftone outputs (0 and 255)
        assert np.any(res1 == 0) and np.any(res1 == 255)
        assert np.any(res2 == 0) and np.any(res2 == 255)



# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestHalftoneFilterValidation:

    def test_size_less_than_two_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="size"):
            HalftoneFilter(params={"size": 1})

    def test_blend_strength_negative_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="blend_strength"):
            HalftoneFilter(params={"blend_strength": -0.1})

    def test_blend_strength_above_one_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="blend_strength"):
            HalftoneFilter(params={"blend_strength": 1.1})

    def test_default_params_succeed(self) -> None:
        filt = HalftoneFilter()
        assert filt._size == 8
        assert filt._angle == 45.0
        assert filt._blend_strength == 0.8
