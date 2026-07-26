"""Unit tests for BlendFilter (filters.blend).

Because the blend operations are deterministic algebraic functions, precise
assertions about expected output values are possible for carefully chosen
inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from comic_renderer.filters.blend import BlendFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid(v: int, h: int = 8, w: int = 8) -> np.ndarray:
    return np.full((h, w, 3), v, dtype=np.uint8)


def _pixel(r: int, g: int, b: int) -> np.ndarray:
    return np.array([[[r, g, b]]], dtype=np.uint8)


def _blend(mode: str, strength: float, image: np.ndarray) -> np.ndarray:
    return BlendFilter(params={"mode": mode, "strength": strength}).apply(image)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestBlendFilterContract:

    @pytest.mark.parametrize("mode", ["overlay", "soft_light", "multiply", "screen"])
    def test_output_is_uint8(self, mode: str) -> None:
        assert _blend(mode, 0.5, _solid(128)).dtype == np.uint8

    @pytest.mark.parametrize("mode", ["overlay", "soft_light", "multiply", "screen"])
    def test_output_has_three_channels(self, mode: str) -> None:
        result = _blend(mode, 0.5, _solid(128))
        assert result.ndim == 3 and result.shape[2] == 3

    @pytest.mark.parametrize("mode", ["overlay", "soft_light", "multiply", "screen"])
    def test_shape_preserved(self, mode: str) -> None:
        image = np.zeros((24, 48, 3), dtype=np.uint8)
        assert _blend(mode, 0.5, image).shape == (24, 48, 3)

    def test_output_values_in_uint8_range(self) -> None:
        result = _blend("overlay", 1.0, _solid(255))
        assert result.min() >= 0 and result.max() <= 255


# ---------------------------------------------------------------------------
# No-op guarantee
# ---------------------------------------------------------------------------


class TestBlendFilterNoop:

    @pytest.mark.parametrize("mode", ["overlay", "soft_light", "multiply", "screen"])
    def test_strength_zero_is_noop(self, mode: str) -> None:
        """strength=0 must return a copy identical to the input."""
        image = _solid(100)
        result = _blend(mode, 0.0, image)
        np.testing.assert_array_equal(result, image)

    def test_returns_copy_not_same_object(self) -> None:
        image = _solid(128)
        result = _blend("overlay", 0.0, image)
        assert result is not image


# ---------------------------------------------------------------------------
# Overlay – known pixel assertions
# ---------------------------------------------------------------------------


class TestBlendFilterOverlay:

    def test_black_stays_black_full_strength(self) -> None:
        """overlay(0, 0) = 0."""
        result = _blend("overlay", 1.0, _solid(0))
        assert np.all(result == 0)

    def test_white_stays_white_full_strength(self) -> None:
        """overlay(255, 255) = 255."""
        result = _blend("overlay", 1.0, _solid(255))
        assert np.all(result == 255)

    def test_midgray_darkens_in_overlay(self) -> None:
        """128 is the darkest point in self-overlay (boundary = 128 exactly,
        goes to multiply branch: 2*128*128/255 = 128.5 → 128 or 129)."""
        result = _blend("overlay", 1.0, _solid(128))
        # overlay of 128 with itself ≈ 128 (boundary is exactly at 128)
        assert abs(int(result[0, 0, 0]) - 128) <= 2

    def test_dark_values_darkened_by_overlay(self) -> None:
        """Below 128, overlay multiplies: result < input."""
        for v in (50, 64, 100):
            # At strength=1: blended = 2*v*v/255; result = blended
            # 2*64*64/255 ≈ 32 < 64
            blended_expected = 2.0 * v * v / 255.0
            result = _blend("overlay", 1.0, _solid(v))
            assert abs(int(result[0, 0, 0]) - blended_expected) <= 2

    def test_light_values_lightened_by_overlay(self) -> None:
        """Above 128, overlay screens: result > input."""
        for v in (160, 200, 230):
            # At strength=1: blended = 255 - 2*(255-v)*(255-v)/255
            blended_expected = 255.0 - 2.0 * (255.0 - v) ** 2 / 255.0
            result = _blend("overlay", 1.0, _solid(v))
            assert abs(int(result[0, 0, 0]) - blended_expected) <= 2

    def test_strength_interpolation(self) -> None:
        """At strength=0.5, result is halfway between original and blend."""
        v = 50
        blended_v = 2.0 * v * v / 255.0
        expected = v * 0.5 + blended_v * 0.5
        result = _blend("overlay", 0.5, _solid(v))
        assert abs(int(result[0, 0, 0]) - expected) <= 2


# ---------------------------------------------------------------------------
# Multiply mode
# ---------------------------------------------------------------------------


class TestBlendFilterMultiply:

    def test_multiply_darkens_image(self) -> None:
        """Self-multiply always produces values <= original."""
        for v in (50, 100, 200):
            result = _blend("multiply", 1.0, _solid(v))
            assert int(result[0, 0, 0]) <= v

    def test_multiply_white_stays_white(self) -> None:
        """255 × 255 / 255 = 255."""
        result = _blend("multiply", 1.0, _solid(255))
        assert np.all(result == 255)

    def test_multiply_black_stays_black(self) -> None:
        """0 × 0 / 255 = 0."""
        result = _blend("multiply", 1.0, _solid(0))
        assert np.all(result == 0)

    def test_known_multiply_value(self) -> None:
        """200 × 200 / 255 ≈ 157."""
        result = _blend("multiply", 1.0, _solid(200))
        assert abs(int(result[0, 0, 0]) - 157) <= 2


# ---------------------------------------------------------------------------
# Screen mode
# ---------------------------------------------------------------------------


class TestBlendFilterScreen:

    def test_screen_lightens_image(self) -> None:
        """Self-screen always produces values >= original."""
        for v in (50, 100, 200):
            result = _blend("screen", 1.0, _solid(v))
            assert int(result[0, 0, 0]) >= v

    def test_screen_black_stays_black(self) -> None:
        """255 - (255-0)*(255-0)/255 = 0."""
        result = _blend("screen", 1.0, _solid(0))
        assert np.all(result == 0)

    def test_screen_white_stays_white(self) -> None:
        result = _blend("screen", 1.0, _solid(255))
        assert np.all(result == 255)

    def test_known_screen_value(self) -> None:
        """255 - (255-100)*(255-100)/255 ≈ 160."""
        result = _blend("screen", 1.0, _solid(100))
        expected = 255.0 - (155.0 * 155.0) / 255.0
        assert abs(int(result[0, 0, 0]) - expected) <= 2


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestBlendFilterValidation:

    def test_unknown_mode_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown mode"):
            BlendFilter(params={"mode": "dodge"})

    def test_strength_above_1_raises(self) -> None:
        with pytest.raises(ValueError, match="strength"):
            BlendFilter(params={"strength": 1.1})

    def test_strength_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="strength"):
            BlendFilter(params={"strength": -0.01})

    def test_default_mode_is_overlay(self) -> None:
        filt = BlendFilter()
        assert filt._mode == "overlay"
        assert filt._strength == 0.5
