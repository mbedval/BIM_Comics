"""Unit tests for ToneCurveFilter (filters.tonecurve)."""

from __future__ import annotations

import numpy as np
import pytest

from comic_renderer.filters.tonecurve import ToneCurveFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid(value: int = 128, h: int = 16, w: int = 16) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _apply(image: np.ndarray, **params) -> np.ndarray:
    return ToneCurveFilter(params=params).apply(image)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestToneCurveFilterContract:

    def test_output_is_uint8(self) -> None:
        assert _apply(_solid()).dtype == np.uint8

    def test_output_has_three_channels(self) -> None:
        result = _apply(_solid())
        assert result.ndim == 3 and result.shape[2] == 3

    def test_shape_preserved(self) -> None:
        image = np.zeros((32, 64, 3), dtype=np.uint8)
        result = _apply(image)
        assert result.shape == (32, 64, 3)

    def test_output_values_in_uint8_range(self) -> None:
        # Extreme tonecurve
        result = _apply(_solid(200), points=[[0, 255], [255, 0]])
        assert result.min() >= 0 and result.max() <= 255


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestToneCurveFilterFunctional:

    def test_identity_curve(self) -> None:
        """Default points [[0, 0], [255, 255]] must act as identity."""
        image = np.array([[[10, 80, 200]]], dtype=np.uint8)
        result = _apply(image)
        np.testing.assert_array_equal(result, image)

    def test_inverse_curve(self) -> None:
        """[[0, 255], [255, 0]] should invert the image colors."""
        image = np.array([[[10, 80, 200]]], dtype=np.uint8)
        expected = np.array([[[245, 175, 55]]], dtype=np.uint8)
        result = _apply(image, points=[[0, 255], [255, 0]])
        np.testing.assert_array_equal(result, expected)

    def test_piecewise_interpolation(self) -> None:
        """Verify linear interpolation works on a specific test point.
        For curve [[0, 0], [100, 50], [255, 255]]:
        Input 50 should yield 25.
        Input 150 should yield: 50 + (150-100)/(255-100) * (255-50) = 50 + 50/155 * 205 ≈ 50 + 66.1 ≈ 116.
        """
        image = np.array([[[50, 150, 0]]], dtype=np.uint8)
        result = _apply(image, points=[[0, 0], [100, 50], [255, 255]])
        assert result[0, 0, 0] == 25
        assert abs(int(result[0, 0, 1]) - 116) <= 1

    def test_returns_copy(self) -> None:
        image = _solid(128)
        result = _apply(image)
        assert result is not image


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestToneCurveFilterValidation:

    def test_points_not_list_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="points must be a list"):
            ToneCurveFilter(params={"points": "invalid"})

    def test_too_few_points_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least 2 points"):
            ToneCurveFilter(params={"points": [[0, 0]]})

    def test_invalid_point_coordinates_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="coordinates must be in"):
            ToneCurveFilter(params={"points": [[0, -5], [255, 255]]})

    def test_point_missing_start_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="first point must start at x=0"):
            ToneCurveFilter(params={"points": [[10, 0], [255, 255]]})

    def test_point_missing_end_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="last point must end at x=255"):
            ToneCurveFilter(params={"points": [[0, 0], [240, 255]]})

    def test_duplicate_x_coordinates_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="duplicate x-coordinate"):
            ToneCurveFilter(params={"points": [[0, 0], [128, 50], [128, 100], [255, 255]]})

    def test_bad_point_format_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="invalid point format"):
            ToneCurveFilter(params={"points": [[0, 0], [128], [255, 255]]})
