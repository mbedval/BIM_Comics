"""Unit tests for TextureFilter (filters.texture)."""

from __future__ import annotations

import numpy as np
import pytest

from comic_renderer.filters.texture import TextureFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid(v: int = 128, h: int = 32, w: int = 32) -> np.ndarray:
    return np.full((h, w, 3), v, dtype=np.uint8)


def _gradient(h: int = 32, w: int = 32) -> np.ndarray:
    row = np.linspace(0, 255, w, dtype=np.uint8)
    gray = np.tile(row, (h, 1))
    return np.stack([gray, gray, gray], axis=2)


def _apply(mode: str = "paper", strength: float = 0.15, seed: int = 42, **kw) -> TextureFilter:
    return TextureFilter(params={"mode": mode, "strength": strength, "seed": seed, **kw})


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestTextureFilterContract:

    @pytest.mark.parametrize("mode", ["paper", "grain"])
    def test_output_is_uint8(self, mode: str) -> None:
        assert _apply(mode).apply(_solid()).dtype == np.uint8

    @pytest.mark.parametrize("mode", ["paper", "grain"])
    def test_output_has_three_channels(self, mode: str) -> None:
        result = _apply(mode).apply(_solid())
        assert result.ndim == 3 and result.shape[2] == 3

    @pytest.mark.parametrize("mode", ["paper", "grain"])
    def test_shape_preserved(self, mode: str) -> None:
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        result = _apply(mode).apply(image)
        assert result.shape == (48, 64, 3)

    def test_output_values_in_uint8_range(self) -> None:
        result = _apply("paper", strength=0.9).apply(_solid(128))
        assert result.min() >= 0 and result.max() <= 255


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestTextureFilterFunctional:

    @pytest.mark.parametrize("mode", ["paper", "grain"])
    def test_strength_zero_is_noop(self, mode: str) -> None:
        """strength=0 must return a copy equal to the input."""
        image = _gradient()
        result = _apply(mode, strength=0.0).apply(image)
        np.testing.assert_array_equal(result, image)

    def test_texture_changes_some_pixels(self) -> None:
        """With non-zero strength, at least some pixels must change."""
        image = _solid(128)
        result = _apply("paper", strength=0.5).apply(image)
        assert not np.array_equal(result, image)

    def test_same_seed_produces_same_result(self) -> None:
        """Identical seeds must produce identical output for the same image."""
        image = _gradient()
        r1 = _apply("paper", strength=0.3, seed=7).apply(image)
        r2 = _apply("paper", strength=0.3, seed=7).apply(image)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_produce_different_results(self) -> None:
        image = _solid(128)
        r1 = _apply("paper", strength=0.5, seed=1).apply(image)
        r2 = _apply("paper", strength=0.5, seed=999).apply(image)
        assert not np.array_equal(r1, r2)

    def test_grain_mode_changes_pixels(self) -> None:
        image = _solid(128)
        result = _apply("grain", strength=0.5).apply(image)
        assert not np.array_equal(result, image)

    def test_higher_strength_more_deviation(self) -> None:
        """Higher strength → larger average deviation from the original."""
        image = _solid(128)
        r_low = _apply("paper", strength=0.1, seed=42).apply(image)
        r_high = _apply("paper", strength=0.8, seed=42).apply(image)
        dev_low = np.abs(r_low.astype(int) - image.astype(int)).mean()
        dev_high = np.abs(r_high.astype(int) - image.astype(int)).mean()
        assert dev_high > dev_low

    def test_grain_size_parameter_accepted(self) -> None:
        result = _apply("grain", strength=0.3, grain_size=5).apply(_solid())
        assert result.dtype == np.uint8

    def test_different_image_sizes(self) -> None:
        """Texture generation must not crash for non-square images."""
        image = np.full((7, 15, 3), 128, dtype=np.uint8)
        result = _apply("paper", strength=0.2).apply(image)
        assert result.shape == (7, 15, 3)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestTextureFilterValidation:

    def test_unknown_mode_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown mode"):
            TextureFilter(params={"mode": "canvas"})

    def test_strength_above_1_raises(self) -> None:
        with pytest.raises(ValueError, match="strength"):
            TextureFilter(params={"strength": 1.1})

    def test_negative_strength_raises(self) -> None:
        with pytest.raises(ValueError, match="strength"):
            TextureFilter(params={"strength": -0.01})

    def test_grain_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="grain_size"):
            TextureFilter(params={"grain_size": 0})

    def test_default_params_do_not_raise(self) -> None:
        filt = TextureFilter()
        assert filt._mode == "paper"
        assert filt._strength == 0.15
        assert filt._seed == 42
