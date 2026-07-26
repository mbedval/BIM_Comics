"""Unit tests for VignetteFilter (filters.vignette)."""

from __future__ import annotations

import numpy as np
import pytest

from comic_renderer.filters.vignette import VignetteFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid(value: int = 128, h: int = 32, w: int = 32) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _apply(image: np.ndarray, **params) -> np.ndarray:
    return VignetteFilter(params=params).apply(image)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestVignetteFilterContract:

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
        result = _apply(_solid(255), strength=1.0)
        assert result.min() >= 0 and result.max() <= 255


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestVignetteFilterFunctional:

    def test_strength_zero_is_noop(self) -> None:
        """strength=0 must return a copy identical to the input."""
        image = _solid(128)
        result = _apply(image, strength=0.0)
        np.testing.assert_array_equal(result, image)

    def test_center_pixel_mostly_unchanged(self) -> None:
        """The center pixel of the image should remain unchanged because the vignette is centered."""
        image = _solid(200, h=33, w=33)  # Odd dimensions so center pixel exists exactly
        result = _apply(image, strength=0.8, radius=1.0)
        cy, cx = 16, 16
        np.testing.assert_array_equal(result[cy, cx], image[cy, cx])

    def test_corners_are_darkened_or_tinted(self) -> None:
        """The corner pixels must be closer to the vignette color than the original color."""
        image = _solid(200, h=32, w=32)
        v_color = [10, 20, 30]
        result = _apply(image, strength=0.8, radius=1.0, color=v_color)
        
        # Corner pixel (0, 0)
        corner_val = result[0, 0].astype(int)
        # It must be darker than 200 and shifted towards [10, 20, 30]
        assert corner_val[0] < 200
        assert corner_val[1] < 200
        assert corner_val[2] < 200

    def test_radius_changes_falloff(self) -> None:
        """A larger radius must result in lighter corner pixels compared to a smaller radius."""
        image = _solid(200, h=32, w=32)
        r_small = _apply(image, strength=0.8, radius=0.5, color=[0, 0, 0])
        r_large = _apply(image, strength=0.8, radius=2.0, color=[0, 0, 0])
        
        # At (0, 0), the large radius should be brighter than the small radius
        assert r_large[0, 0, 0] > r_small[0, 0, 0]

    def test_returns_copy(self) -> None:
        image = _solid(128)
        result = _apply(image)
        assert result is not image

    def test_cache_invalidation_on_shape_change(self) -> None:
        """Applying the same filter instance to different shapes must invalidate cache and work correctly."""
        filt = VignetteFilter(params={"strength": 0.8, "radius": 1.0, "color": [0, 0, 0]})
        img1 = _solid(200, h=16, w=16)
        img2 = _solid(200, h=32, w=32)

        res1 = filt.apply(img1)
        res2 = filt.apply(img2)

        assert res1.shape == (16, 16, 3)
        assert res2.shape == (32, 32, 3)
        # Verify corners are shaded on both
        assert res1[0, 0, 0] < 200
        assert res2[0, 0, 0] < 200



# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestVignetteFilterValidation:

    def test_negative_strength_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="strength"):
            VignetteFilter(params={"strength": -0.1})

    def test_strength_above_one_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="strength"):
            VignetteFilter(params={"strength": 1.05})

    def test_radius_too_small_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="radius"):
            VignetteFilter(params={"radius": 0.05})

    def test_invalid_color_format_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="color must be a three-element list"):
            VignetteFilter(params={"color": [255, 255]})

    def test_color_out_of_bounds_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="color components must be in"):
            VignetteFilter(params={"color": [300, 0, 0]})

    def test_default_params_succeed(self) -> None:
        filt = VignetteFilter()
        assert filt._strength == 0.5
        assert filt._radius == 1.0
        np.testing.assert_array_equal(filt._color, [0.0, 0.0, 0.0])
