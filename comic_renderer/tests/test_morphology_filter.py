"""Unit tests for MorphologyFilter (filters.morphology).

Morphological operations have well-defined, deterministic effects on
structurally simple images (solid white, solid black, single-pixel bright
dots), enabling exact pixel-level assertions.
"""

from __future__ import annotations

import numpy as np
import pytest

from comic_renderer.filters.morphology import MorphologyFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid(v: int, h: int = 16, w: int = 16) -> np.ndarray:
    return np.full((h, w, 3), v, dtype=np.uint8)


def _spot_image(h: int = 16, w: int = 16, spot_v: int = 255, bg_v: int = 0) -> np.ndarray:
    """Return a dark image with a single bright pixel at the centre."""
    img = np.full((h, w, 3), bg_v, dtype=np.uint8)
    img[h // 2, w // 2, :] = spot_v
    return img


def _apply(operation: str = "close", kernel_size: int = 3, **kw) -> MorphologyFilter:
    return MorphologyFilter(params={"operation": operation, "kernel_size": kernel_size, **kw})


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestMorphologyFilterContract:

    @pytest.mark.parametrize("op", ["dilate", "erode", "close", "open"])
    def test_output_is_uint8(self, op: str) -> None:
        assert _apply(op).apply(_solid(128)).dtype == np.uint8

    @pytest.mark.parametrize("op", ["dilate", "erode", "close", "open"])
    def test_output_has_three_channels(self, op: str) -> None:
        result = _apply(op).apply(_solid(128))
        assert result.ndim == 3 and result.shape[2] == 3

    @pytest.mark.parametrize("op", ["dilate", "erode", "close", "open"])
    def test_shape_preserved(self, op: str) -> None:
        image = np.zeros((24, 48, 3), dtype=np.uint8)
        assert _apply(op).apply(image).shape == (24, 48, 3)


# ---------------------------------------------------------------------------
# Structural / semantic tests
# ---------------------------------------------------------------------------


class TestMorphologyFilterSemantic:

    def test_dilate_expands_bright_spot(self) -> None:
        """Dilation of a single bright pixel must produce a neighbourhood of
        bright pixels (at least the kernel area must be non-zero)."""
        image = _spot_image()
        result = _apply("dilate", kernel_size=3, kernel_shape="rect").apply(image)
        h, w = image.shape[:2]
        cy, cx = h // 2, w // 2
        # The 3×3 neighbourhood around the spot must be all bright.
        patch = result[cy - 1 : cy + 2, cx - 1 : cx + 2, :]
        assert np.all(patch == 255), (
            f"Expected all-255 in 3x3 patch after dilate, got:\n{patch[:, :, 0]}"
        )

    def test_erode_removes_isolated_spot(self) -> None:
        """Erosion of a single bright pixel in a dark background must remove it."""
        image = _spot_image()
        result = _apply("erode", kernel_size=3).apply(image)
        h, w = image.shape[:2]
        cy, cx = h // 2, w // 2
        assert result[cy, cx, 0] == 0

    def test_dilate_solid_white_unchanged(self) -> None:
        """Dilation of a fully-white image leaves it unchanged."""
        image = _solid(255)
        result = _apply("dilate").apply(image)
        np.testing.assert_array_equal(result, image)

    def test_erode_solid_black_unchanged(self) -> None:
        """Erosion of a fully-black image leaves it unchanged."""
        image = _solid(0)
        result = _apply("erode").apply(image)
        np.testing.assert_array_equal(result, image)

    def test_close_fills_small_dark_gap(self) -> None:
        """Close should fill small dark gaps in a bright region."""
        # Build a mostly-white image with a single dark pixel in the centre.
        image = _solid(255)
        h, w = image.shape[:2]
        image[h // 2, w // 2, :] = 0  # single dark spot
        result = _apply("close", kernel_size=3).apply(image)
        # After closing, the dark spot should be filled.
        assert result[h // 2, w // 2, 0] == 255

    def test_open_removes_isolated_bright_spot(self) -> None:
        """Open should eliminate an isolated bright pixel smaller than the kernel."""
        image = _spot_image()
        result = _apply("open", kernel_size=3).apply(image)
        h, w = image.shape[:2]
        # After open, the spot (smaller than 3x3 kernel) must be gone.
        assert result[h // 2, w // 2, 0] == 0

    def test_larger_kernel_dilates_more(self) -> None:
        """Larger kernel → larger bright region after dilation."""
        image = _spot_image()
        r_small = _apply("dilate", kernel_size=3).apply(image)
        r_large = _apply("dilate", kernel_size=7).apply(image)
        # More pixels should be bright with a larger kernel.
        assert np.sum(r_large > 0) >= np.sum(r_small > 0)

    def test_multiple_iterations_dilate_more(self) -> None:
        """More iterations → greater dilation."""
        image = _spot_image()
        r1 = _apply("dilate", kernel_size=3, iterations=1).apply(image)
        r3 = _apply("dilate", kernel_size=3, iterations=3).apply(image)
        assert np.sum(r3 > 0) >= np.sum(r1 > 0)

    @pytest.mark.parametrize("shape", ["rect", "ellipse", "cross"])
    def test_all_kernel_shapes_accepted(self, shape: str) -> None:
        result = _apply("dilate", kernel_size=3, kernel_shape=shape).apply(_spot_image())
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestMorphologyFilterValidation:

    def test_unknown_operation_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown operation"):
            MorphologyFilter(params={"operation": "gradient"})

    def test_kernel_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="kernel_size"):
            MorphologyFilter(params={"kernel_size": 0})

    def test_negative_kernel_size_raises(self) -> None:
        with pytest.raises(ValueError, match="kernel_size"):
            MorphologyFilter(params={"kernel_size": -1})

    def test_iterations_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="iterations"):
            MorphologyFilter(params={"iterations": 0})

    def test_unknown_kernel_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown kernel_shape"):
            MorphologyFilter(params={"kernel_shape": "circle"})

    def test_default_params_do_not_raise(self) -> None:
        filt = MorphologyFilter()
        assert filt._operation == "close"
        assert filt._kernel_size == 3
        assert filt._iterations == 1
        assert filt._kernel_shape == "ellipse"
