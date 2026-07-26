"""Unit tests for EdgeFilter (filters.edge)."""

from __future__ import annotations

import numpy as np
import pytest

from comic_renderer.filters.edge import EdgeFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid(v: int, h: int = 16, w: int = 16) -> np.ndarray:
    return np.full((h, w, 3), v, dtype=np.uint8)


def _step_image(h: int = 32, w: int = 64) -> np.ndarray:
    """Image with a sharp vertical edge at the centre."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, : w // 2, :] = 50
    img[:, w // 2 :, :] = 200
    return img


def _apply(method: str = "canny", blend_strength: float = 1.0, **extra) -> EdgeFilter:
    params = {"method": method, "blend_strength": blend_strength, **extra}
    return EdgeFilter(params=params)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestEdgeFilterContract:

    @pytest.mark.parametrize("method", ["canny", "sobel", "laplacian"])
    def test_output_is_uint8(self, method: str) -> None:
        result = _apply(method).apply(_step_image())
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("method", ["canny", "sobel", "laplacian"])
    def test_output_has_three_channels(self, method: str) -> None:
        result = _apply(method).apply(_step_image())
        assert result.ndim == 3 and result.shape[2] == 3

    @pytest.mark.parametrize("method", ["canny", "sobel", "laplacian"])
    def test_shape_preserved(self, method: str) -> None:
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        result = _apply(method).apply(image)
        assert result.shape == (48, 64, 3)

    def test_output_values_in_uint8_range(self) -> None:
        result = _apply("canny", blend_strength=1.0).apply(_step_image())
        assert result.min() >= 0 and result.max() <= 255


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestEdgeFilterFunctional:

    def test_blend_strength_zero_is_noop(self) -> None:
        """blend_strength=0 → output must equal input exactly."""
        image = _step_image()
        result = _apply("canny", blend_strength=0.0).apply(image)
        np.testing.assert_array_equal(result, image)

    @pytest.mark.parametrize("method", ["canny", "sobel", "laplacian"])
    def test_blend_strength_zero_is_noop_all_methods(self, method: str) -> None:
        image = _step_image()
        result = _apply(method, blend_strength=0.0).apply(image)
        np.testing.assert_array_equal(result, image)

    def test_edges_darken_pixels_at_edge(self) -> None:
        """Pixels near the edge must be darker after applying EdgeFilter."""
        image = _step_image()
        result = _apply("canny", blend_strength=1.0).apply(image)
        mid = image.shape[1] // 2

        # In the immediate vicinity of the edge, at least some pixels darkened.
        original_mean = float(image[:, mid - 2 : mid + 2, :].mean())
        result_mean = float(result[:, mid - 2 : mid + 2, :].mean())
        assert result_mean <= original_mean

    def test_uniform_image_unchanged_by_canny(self) -> None:
        """A perfectly uniform image has no edges; output must equal input."""
        image = _solid(128)
        result = _apply("canny", blend_strength=1.0).apply(image)
        np.testing.assert_array_equal(result, image)

    def test_blur_radius_does_not_raise(self) -> None:
        result = _apply("canny", blend_strength=0.8, blur_radius=2).apply(_step_image())
        assert result.dtype == np.uint8

    def test_higher_blend_strength_darker_edges(self) -> None:
        """Increasing blend_strength should produce darker or equal edge pixels."""
        image = _step_image()
        mid = image.shape[1] // 2
        r_low = _apply("sobel", blend_strength=0.3).apply(image)
        r_high = _apply("sobel", blend_strength=1.0).apply(image)
        # The edge region overall should be equal or darker with higher strength.
        assert float(r_high[:, mid - 2 : mid + 2, :].mean()) <= float(
            r_low[:, mid - 2 : mid + 2, :].mean()
        )

    def test_result_is_a_copy(self) -> None:
        image = _step_image()
        result = _apply("canny", blend_strength=0.0).apply(image)
        assert result is not image


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestEdgeFilterValidation:

    def test_unknown_method_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown method"):
            EdgeFilter(params={"method": "watershed"})

    def test_negative_low_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            EdgeFilter(params={"low_threshold": -1, "high_threshold": 100})

    def test_low_gte_high_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="low_threshold"):
            EdgeFilter(params={"low_threshold": 150, "high_threshold": 100})

    def test_equal_thresholds_raises(self) -> None:
        with pytest.raises(ValueError, match="low_threshold"):
            EdgeFilter(params={"low_threshold": 100, "high_threshold": 100})

    def test_even_ksize_raises(self) -> None:
        with pytest.raises(ValueError, match="odd"):
            EdgeFilter(params={"ksize": 4})

    def test_zero_ksize_raises(self) -> None:
        with pytest.raises(ValueError, match="ksize"):
            EdgeFilter(params={"ksize": 0})

    def test_blend_strength_above_1_raises(self) -> None:
        with pytest.raises(ValueError, match="blend_strength"):
            EdgeFilter(params={"blend_strength": 1.1})

    def test_blend_strength_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="blend_strength"):
            EdgeFilter(params={"blend_strength": -0.01})

    def test_negative_blur_radius_raises(self) -> None:
        with pytest.raises(ValueError, match="blur_radius"):
            EdgeFilter(params={"blur_radius": -1})

    def test_default_params_do_not_raise(self) -> None:
        filt = EdgeFilter()
        assert filt._method == "canny"
        assert filt._blend_strength == 1.0
