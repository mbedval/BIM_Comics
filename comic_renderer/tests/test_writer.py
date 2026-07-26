"""Unit tests for ImageWriter (io.writer)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from comic_renderer.config.settings import IOConfig
from comic_renderer.io.writer import ImageWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_io_config(
    story_name: str,
    tmp_path: Path,
    overwrite: bool = False,
) -> IOConfig:
    story_path = tmp_path / "stories" / story_name
    story_path.mkdir(parents=True, exist_ok=True)
    output_root = tmp_path / "output"
    return IOConfig(
        story_path=story_path,
        output_root=output_root,
        overwrite=overwrite,
    )


def _solid_rgb(r: int, g: int, b: int, h: int = 8, w: int = 8) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = [r, g, b]
    return img


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestImageWriter:
    """Tests for ImageWriter.write()."""

    def test_creates_output_file(self, tmp_path: Path) -> None:
        """write() must create the output PNG file."""
        cfg = _make_io_config("story1", tmp_path)
        writer = ImageWriter(io_config=cfg)
        source = cfg.story_path / "001.jpg"
        source.touch()

        image = _solid_rgb(100, 150, 200)
        output_path = writer.write(image, source)

        assert output_path.exists()
        assert output_path.suffix == ".png"

    def test_output_is_png(self, tmp_path: Path) -> None:
        """The written file must be a valid PNG regardless of source extension."""
        cfg = _make_io_config("story1", tmp_path)
        writer = ImageWriter(io_config=cfg)
        source = cfg.story_path / "shot.bmp"
        source.touch()

        output_path = writer.write(_solid_rgb(10, 20, 30), source)

        # Pillow should be able to open it as PNG
        with Image.open(output_path) as img:
            assert img.format == "PNG"

    def test_output_filename_stem_matches_source(self, tmp_path: Path) -> None:
        """Output file has the same stem as the source, with .png extension."""
        cfg = _make_io_config("story1", tmp_path)
        writer = ImageWriter(io_config=cfg)
        source = cfg.story_path / "frame_042.tiff"
        source.touch()

        output_path = writer.write(_solid_rgb(0, 0, 0), source)

        assert output_path.stem == "frame_042"
        assert output_path.suffix == ".png"

    def test_output_directory_is_created_automatically(self, tmp_path: Path) -> None:
        """Parent directories must be created automatically (mkdir -p)."""
        cfg = _make_io_config("deep_story", tmp_path)
        writer = ImageWriter(io_config=cfg)
        source = cfg.story_path / "001.jpg"
        source.touch()

        output_path = writer.write(_solid_rgb(0, 0, 0), source)

        assert output_path.parent.is_dir()

    def test_output_goes_to_output_story_dir(self, tmp_path: Path) -> None:
        """Output file is placed inside output_root/<story_name>/."""
        cfg = _make_io_config("my_story", tmp_path)
        writer = ImageWriter(io_config=cfg)
        source = cfg.story_path / "001.jpg"
        source.touch()

        output_path = writer.write(_solid_rgb(0, 0, 0), source)

        assert output_path.parent == cfg.output_story_dir()

    def test_skip_when_overwrite_false_and_file_exists(self, tmp_path: Path) -> None:
        """When overwrite=False an existing output is NOT replaced."""
        cfg = _make_io_config("story1", tmp_path, overwrite=False)
        writer = ImageWriter(io_config=cfg)
        source = cfg.story_path / "001.jpg"
        source.touch()

        # Write once to create the file
        first_path = writer.write(_solid_rgb(255, 0, 0), source)
        mtime_before = first_path.stat().st_mtime

        # Write again – should be skipped
        writer.write(_solid_rgb(0, 0, 255), source)
        mtime_after = first_path.stat().st_mtime

        assert mtime_before == mtime_after  # File was NOT touched

    def test_overwrite_replaces_existing_file(self, tmp_path: Path) -> None:
        """When overwrite=True the existing output IS replaced."""
        cfg = _make_io_config("story1", tmp_path, overwrite=True)
        writer = ImageWriter(io_config=cfg)
        source = cfg.story_path / "001.jpg"
        source.touch()

        original_color = _solid_rgb(255, 0, 0)
        new_color = _solid_rgb(0, 0, 255)

        writer.write(original_color, source)
        output_path = writer.write(new_color, source)

        # The pixel content should now reflect new_color
        loaded = np.array(Image.open(output_path).convert("RGB"))
        assert loaded[0, 0, 2] > 200  # Blue channel dominant
