"""Unit tests for the image loader (io.loader).

A real 4×4 pixel image is created in-memory and written to a tmp_path file,
then loaded back.  This avoids depending on fixture images while still
exercising the full OpenCV + EXIF path.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from comic_renderer.io.loader import load_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_rgb_png(path: Path, array: np.ndarray) -> None:
    """Save an RGB uint8 array to *path* as PNG using Pillow."""
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(str(path))


def _make_gradient_image(h: int = 16, w: int = 16) -> np.ndarray:
    """Return a simple H×W×3 uint8 gradient (values vary across pixels)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)   # R varies left→right
    img[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8).reshape(h, 1)  # G top→bottom
    img[:, :, 2] = 128
    return img


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadImage:
    """Tests for load_image()."""

    def test_returns_rgb_uint8_array(self, tmp_path: Path) -> None:
        """Loaded image must be an RGB uint8 ndarray."""
        original = _make_gradient_image()
        img_path = tmp_path / "test.png"
        _write_rgb_png(img_path, original)

        result = load_image(img_path)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.ndim == 3
        assert result.shape[2] == 3  # 3 channels

    def test_preserves_dimensions(self, tmp_path: Path) -> None:
        """Width and height must be unchanged after loading."""
        original = _make_gradient_image(h=32, w=64)
        img_path = tmp_path / "dims.png"
        _write_rgb_png(img_path, original)

        result = load_image(img_path)

        assert result.shape[:2] == (32, 64)

    def test_pixel_values_are_preserved(self, tmp_path: Path) -> None:
        """Pixel values must survive a round-trip save → load."""
        original = _make_gradient_image(h=8, w=8)
        img_path = tmp_path / "values.png"
        _write_rgb_png(img_path, original)

        result = load_image(img_path)

        # PNG is lossless so values should be exactly equal.
        np.testing.assert_array_equal(result, original)

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """FileNotFoundError is raised for non-existent paths."""
        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "ghost.jpg")

    def test_corrupt_file_raises_value_error(self, tmp_path: Path) -> None:
        """A non-image file raises ValueError (OpenCV cannot decode it)."""
        corrupt = tmp_path / "corrupt.jpg"
        corrupt.write_bytes(b"this is not a JPEG")

        with pytest.raises(ValueError, match="OpenCV could not decode"):
            load_image(corrupt)

    def test_supports_jpeg_extension(self, tmp_path: Path) -> None:
        """JPEG files can be loaded (lossy, so only check shape/dtype)."""
        original = _make_gradient_image(h=16, w=16)
        img_path = tmp_path / "photo.jpg"
        Image.fromarray(original, mode="RGB").save(str(img_path), quality=95)

        result = load_image(img_path)

        assert result.shape == (16, 16, 3)
        assert result.dtype == np.uint8
