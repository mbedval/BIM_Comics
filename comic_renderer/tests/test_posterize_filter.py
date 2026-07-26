"""Unit tests for PosterizeFilter (filters.posterize).

Posterization is deterministic and algebraically simple, so pixel-perfect
assertions are possible.
"""

from __future__ import annotations

import numpy as np
import pytest

from comic_renderer.filters.posterize import PosterizeFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid(v: int, h: int = 8, w: int = 8) -> np.ndarray:
    """Return a solid single-value RGB image."""
    return np.full((h, w, 3), v, dtype=np.uint8)


def _posterize(levels: int, image: np.ndarray) -> np.ndarray:
    return PosterizeFilter(params={"levels": levels}).apply(image)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestPosterizeFilterContract:

    def test_output_is_uint8(self) -> None:
        assert _posterize(4, _solid(100)).dtype == np.uint8

    def test_output_has_three_channels(self) -> None:
        result = _posterize(4, _solid(100))
        assert result.ndim == 3 and result.shape[2] == 3

    def test_shape_preserved(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        assert _posterize(4, image).shape == (20, 30, 3)

    def test_output_values_in_uint8_range(self) -> None:
        image = np.full((8, 8, 3), 255, dtype=np.uint8)
        result = _posterize(4, image)
        assert result.min() >= 0 and result.max() <= 255


# ---------------------------------------------------------------------------
# Pixel-exact quantization tests  (step = 256 // levels)
# ---------------------------------------------------------------------------

# levels=4 → step=64 → buckets: 0, 64, 128, 192
_L4_CASES: list[tuple[int, int]] = [
    (0,   0),
    (1,   0),
    (63,  0),
    (64,  64),
    (127, 64),
    (128, 128),
    (191, 128),
    (192, 192),
    (254, 192),
    (255, 192),
]

# levels=2 → step=128 → buckets: 0, 128
_L2_CASES: list[tuple[int, int]] = [
    (0,   0),
    (127, 0),
    (128, 128),
    (255, 128),
]

# levels=8 → step=32 → buckets: 0,32,64,96,128,160,192,224
_L8_CASES: list[tuple[int, int]] = [
    (0,   0),
    (31,  0),
    (32,  32),
    (63,  32),
    (64,  64),
    (255, 224),
]


class TestPosterizeFilterQuantization:

    @pytest.mark.parametrize("pixel,expected", _L4_CASES)
    def test_levels_4(self, pixel: int, expected: int) -> None:
        result = _posterize(4, _solid(pixel))
        assert int(result[0, 0, 0]) == expected, (
            f"levels=4, pixel={pixel} → expected {expected}, got {result[0,0,0]}"
        )

    @pytest.mark.parametrize("pixel,expected", _L2_CASES)
    def test_levels_2(self, pixel: int, expected: int) -> None:
        result = _posterize(2, _solid(pixel))
        assert int(result[0, 0, 0]) == expected

    @pytest.mark.parametrize("pixel,expected", _L8_CASES)
    def test_levels_8(self, pixel: int, expected: int) -> None:
        result = _posterize(8, _solid(pixel))
        assert int(result[0, 0, 0]) == expected

    def test_number_of_distinct_values_per_channel(self) -> None:
        """Output must not contain more distinct values than expected.

        When 256 is not divisible by ``levels``, the step size
        ``step = 256 // levels`` may create up to ``levels + 1`` distinct
        output values.  We only test levels that evenly divide 256 so the
        assertion can be exact.
        """
        image = np.arange(256, dtype=np.uint8).reshape(1, 256, 1)
        image = np.repeat(image, 3, axis=2)  # (1, 256, 3)
        for levels in (2, 4, 8, 16, 32, 64):
            result = _posterize(levels, image)
            distinct = len(np.unique(result[:, :, 0]))
            assert distinct <= levels, (
                f"levels={levels}: expected <= {levels} distinct values, got {distinct}"
            )

    def test_levels_256_is_near_identity(self) -> None:
        """With 256 levels (step=1) output should equal input."""
        image = np.arange(256, dtype=np.uint8).reshape(1, 256, 1)
        image = np.repeat(image, 3, axis=2)
        result = _posterize(256, image)
        np.testing.assert_array_equal(result, image)

    def test_colour_channels_posterized_independently(self) -> None:
        """Each colour channel must be quantized on its own."""
        pixel = np.array([[[100, 128, 200]]], dtype=np.uint8)
        # levels=4 → step=64
        result = _posterize(4, pixel)
        assert result[0, 0, 0] == 64   # R: 100 // 64 * 64
        assert result[0, 0, 1] == 128  # G: 128 // 64 * 64
        assert result[0, 0, 2] == 192  # B: 200 // 64 * 64


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestPosterizeFilterValidation:

    def test_levels_1_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="levels"):
            PosterizeFilter(params={"levels": 1})

    def test_levels_0_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="levels"):
            PosterizeFilter(params={"levels": 0})

    def test_levels_257_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="levels"):
            PosterizeFilter(params={"levels": 257})

    def test_default_levels_is_4(self) -> None:
        filt = PosterizeFilter()
        assert filt._levels == 4
