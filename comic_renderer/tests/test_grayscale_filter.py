"""Unit tests for GrayscaleFilter (filters.grayscale)."""

from __future__ import annotations

import numpy as np
import pytest

from comic_renderer.filters.grayscale import GrayscaleFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid(r: int, g: int, b: int, h: int = 8, w: int = 8) -> np.ndarray:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = [r, g, b]
    return arr


def _apply(method: str, image: np.ndarray) -> np.ndarray:
    return GrayscaleFilter(params={"method": method}).apply(image)


# ---------------------------------------------------------------------------
# Contract tests (common to all methods)
# ---------------------------------------------------------------------------


class TestGrayscaleFilterContract:
    """Output must always be (H, W, 3) uint8 with R = G = B."""

    @pytest.mark.parametrize("method", ["luminance", "average", "lightness"])
    def test_output_is_uint8(self, method: str) -> None:
        assert _apply(method, _solid(100, 150, 200)).dtype == np.uint8

    @pytest.mark.parametrize("method", ["luminance", "average", "lightness"])
    def test_output_has_three_channels(self, method: str) -> None:
        result = _apply(method, _solid(100, 150, 200))
        assert result.ndim == 3 and result.shape[2] == 3

    @pytest.mark.parametrize("method", ["luminance", "average", "lightness"])
    def test_output_channels_are_equal(self, method: str) -> None:
        """All three output channels must carry the same grayscale value."""
        result = _apply(method, _solid(80, 120, 200))
        np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 1], result[:, :, 2])

    @pytest.mark.parametrize("method", ["luminance", "average", "lightness"])
    def test_shape_preserved(self, method: str) -> None:
        image = np.zeros((32, 64, 3), dtype=np.uint8)
        result = _apply(method, image)
        assert result.shape == (32, 64, 3)

    @pytest.mark.parametrize("method", ["luminance", "average", "lightness"])
    def test_pure_black_stays_black(self, method: str) -> None:
        result = _apply(method, _solid(0, 0, 0))
        assert np.all(result == 0)

    @pytest.mark.parametrize("method", ["luminance", "average", "lightness"])
    def test_pure_white_stays_white(self, method: str) -> None:
        result = _apply(method, _solid(255, 255, 255))
        assert np.all(result == 255)

    @pytest.mark.parametrize("method", ["luminance", "average", "lightness"])
    def test_already_gray_input_unchanged(self, method: str) -> None:
        """A solid gray (R=G=B=v) must produce the same value v in all channels."""
        for v in [64, 128, 200]:
            result = _apply(method, _solid(v, v, v))
            # Allow ±1 for rounding in colour-space conversions.
            assert abs(int(result[0, 0, 0]) - v) <= 1


# ---------------------------------------------------------------------------
# Luminance method
# ---------------------------------------------------------------------------


class TestGrayscaleLuminance:
    """BT.601: L = 0.299·R + 0.587·G + 0.114·B."""

    def test_pure_red_matches_bt601(self) -> None:
        # 0.299 * 255 ≈ 76
        result = _apply("luminance", _solid(255, 0, 0))
        assert abs(int(result[0, 0, 0]) - 76) <= 1

    def test_pure_green_matches_bt601(self) -> None:
        # 0.587 * 255 ≈ 150
        result = _apply("luminance", _solid(0, 255, 0))
        assert abs(int(result[0, 0, 0]) - 150) <= 1

    def test_pure_blue_matches_bt601(self) -> None:
        # 0.114 * 255 ≈ 29
        result = _apply("luminance", _solid(0, 0, 255))
        assert abs(int(result[0, 0, 0]) - 29) <= 1

    def test_default_method_is_luminance(self) -> None:
        f = GrayscaleFilter()
        result = f.apply(_solid(255, 0, 0))
        assert abs(int(result[0, 0, 0]) - 76) <= 1


# ---------------------------------------------------------------------------
# Average method
# ---------------------------------------------------------------------------


class TestGrayscaleAverage:
    def test_known_average(self) -> None:
        # (90 + 60 + 30) / 3 = 60
        result = _apply("average", _solid(90, 60, 30))
        assert abs(int(result[0, 0, 0]) - 60) <= 1

    def test_average_is_symmetric(self) -> None:
        """Swapping channels must not change the average."""
        v1 = int(_apply("average", _solid(10, 100, 200))[0, 0, 0])
        v2 = int(_apply("average", _solid(200, 10, 100))[0, 0, 0])
        assert abs(v1 - v2) <= 1


# ---------------------------------------------------------------------------
# Lightness method
# ---------------------------------------------------------------------------


class TestGrayscaleLightness:
    def test_known_lightness(self) -> None:
        # (max(100, 50, 200) + min(100, 50, 200)) / 2 = (200 + 50) / 2 = 125
        result = _apply("lightness", _solid(100, 50, 200))
        assert abs(int(result[0, 0, 0]) - 125) <= 1

    def test_lightness_of_saturated_color(self) -> None:
        # max=255, min=0 → 127
        result = _apply("lightness", _solid(255, 0, 0))
        assert abs(int(result[0, 0, 0]) - 127) <= 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestGrayscaleValidation:
    def test_invalid_method_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown method"):
            GrayscaleFilter(params={"method": "nonsense"})
