"""Unit tests for SharpenFilter (filters.sharpen)."""

from __future__ import annotations

import numpy as np
import pytest

from comic_renderer.filters.sharpen import SharpenFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform(value: int = 128, h: int = 16, w: int = 16) -> np.ndarray:
    """Return a solid uniform RGB image."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _step_image(h: int = 16, w: int = 32) -> np.ndarray:
    """Return an image with a hard vertical edge at the centre.

    Left half = 50, right half = 200.  Sharpening should widen the contrast
    band around the edge (more extreme values near the boundary).
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, : w // 2, :] = 50
    img[:, w // 2 :, :] = 200
    return img


def _sharpen(strength: float, image: np.ndarray, **kw) -> np.ndarray:
    params = {"strength": strength, **kw}
    return SharpenFilter(params=params).apply(image)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestSharpenFilterContract:

    def test_output_is_uint8(self) -> None:
        assert _sharpen(1.0, _uniform()).dtype == np.uint8

    def test_output_has_three_channels(self) -> None:
        result = _sharpen(1.0, _uniform())
        assert result.ndim == 3 and result.shape[2] == 3

    def test_shape_preserved(self) -> None:
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        assert _sharpen(1.0, image).shape == (48, 64, 3)

    def test_output_values_in_uint8_range(self) -> None:
        result = _sharpen(2.0, _step_image())
        assert result.min() >= 0 and result.max() <= 255


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestSharpenFilterFunctional:

    def test_strength_zero_preserves_uniform_image(self) -> None:
        """strength=0 + uniform image → output must equal input (no Gaussian effect)."""
        image = _uniform(128)
        result = _sharpen(0.0, image)
        # For a uniform image, blurred == image, so addWeighted gives image.
        np.testing.assert_array_equal(result, image)

    def test_any_strength_preserves_uniform_image(self) -> None:
        """For a uniform image, sharpening has no effect regardless of strength."""
        for strength in (0.5, 1.0, 2.0, 3.0):
            image = _uniform(100)
            result = _sharpen(strength, image)
            np.testing.assert_array_equal(result, image)

    def test_sharpening_increases_edge_contrast(self) -> None:
        """Edge pixels should move further from the background after sharpening."""
        image = _step_image()
        sharpened = _sharpen(2.0, image)

        # The pixel just right of the edge (col = w//2) should be brighter
        # than the original 200 (overshooting), OR the pixel just left should
        # be darker than 50.  Check max minus min across the edge area.
        h, w = image.shape[:2]
        mid = w // 2

        original_range = (
            int(image[:, mid, 0].max()) - int(image[:, mid - 1, 0].min())
        )
        sharpened_range = (
            int(sharpened[:, mid, 0].max()) - int(sharpened[:, mid - 1, 0].min())
        )
        # Sharpening should never *reduce* the range at an edge.
        assert sharpened_range >= original_range

    def test_returns_copy_not_same_object(self) -> None:
        image = _uniform(128)
        result = _sharpen(1.0, image)
        assert result is not image

    def test_custom_kernel_size_accepted(self) -> None:
        result = SharpenFilter(params={"strength": 1.0, "kernel_size": 3}).apply(_uniform())
        assert result.dtype == np.uint8

    def test_sigma_affects_output(self) -> None:
        """Different sigma values should produce different results on a step image."""
        image = _step_image()
        r1 = SharpenFilter(params={"strength": 1.5, "sigma": 0.5}).apply(image)
        r2 = SharpenFilter(params={"strength": 1.5, "sigma": 3.0}).apply(image)
        assert not np.array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestSharpenFilterValidation:

    def test_negative_strength_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="strength"):
            SharpenFilter(params={"strength": -0.1})

    def test_even_kernel_size_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="odd"):
            SharpenFilter(params={"strength": 1.0, "kernel_size": 4})

    def test_kernel_size_less_than_3_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="kernel_size"):
            SharpenFilter(params={"strength": 1.0, "kernel_size": 1})

    def test_zero_sigma_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="sigma"):
            SharpenFilter(params={"strength": 1.0, "sigma": 0.0})

    def test_negative_sigma_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="sigma"):
            SharpenFilter(params={"strength": 1.0, "sigma": -1.0})

    def test_default_params_do_not_raise(self) -> None:
        filt = SharpenFilter()
        assert filt._strength == 1.0
        assert filt._kernel_size == 5
        assert filt._sigma == 1.0
