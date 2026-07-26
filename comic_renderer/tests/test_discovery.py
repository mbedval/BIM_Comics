"""Unit tests for StoryDiscovery (io.discovery).

These tests are self-contained: they use tmp_path to create ephemeral
directories and never touch the real images/ folder.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from comic_renderer.config.settings import IOConfig
from comic_renderer.io.discovery import StoryDiscovery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_io_config(story_path: Path, output_root: Path | None = None) -> IOConfig:
    return IOConfig(
        story_path=story_path,
        output_root=output_root or (story_path.parent / "output"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStoryDiscovery:
    """Tests for StoryDiscovery.discover()."""

    def test_discovers_supported_images(self, tmp_path: Path) -> None:
        """Supported extensions are returned; unsupported files are ignored."""
        story = tmp_path / "story1"
        story.mkdir()
        (story / "001.jpg").touch()
        (story / "002.png").touch()
        (story / "003.bmp").touch()
        (story / "readme.txt").touch()  # Must be ignored
        (story / "thumbs.db").touch()   # Must be ignored

        discovery = StoryDiscovery(_make_io_config(story))
        result = discovery.discover()

        assert len(result) == 3
        assert all(p.suffix.lower() in {".jpg", ".png", ".bmp"} for p in result)

    def test_result_is_sorted_by_name(self, tmp_path: Path) -> None:
        """Images are returned in ascending lexicographic order."""
        story = tmp_path / "story1"
        story.mkdir()
        (story / "003.jpg").touch()
        (story / "001.jpg").touch()
        (story / "002.jpg").touch()

        discovery = StoryDiscovery(_make_io_config(story))
        result = discovery.discover()

        names = [p.name for p in result]
        assert names == sorted(names)

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        """An empty story directory returns an empty list (no exception)."""
        story = tmp_path / "empty_story"
        story.mkdir()

        discovery = StoryDiscovery(_make_io_config(story))
        assert discovery.discover() == []

    def test_missing_directory_raises_file_not_found(self, tmp_path: Path) -> None:
        """A non-existent story_path raises FileNotFoundError."""
        story = tmp_path / "does_not_exist"
        discovery = StoryDiscovery(_make_io_config(story))

        with pytest.raises(FileNotFoundError, match="does not exist"):
            discovery.discover()

    def test_non_directory_path_raises_file_not_found(self, tmp_path: Path) -> None:
        """A regular file passed as story_path raises FileNotFoundError."""
        file_path = tmp_path / "not_a_dir.jpg"
        file_path.touch()

        discovery = StoryDiscovery(_make_io_config(file_path))
        with pytest.raises(FileNotFoundError, match="not a directory"):
            discovery.discover()

    def test_tiff_and_webp_are_supported(self, tmp_path: Path) -> None:
        """tiff, tif, and webp extensions are included."""
        story = tmp_path / "story2"
        story.mkdir()
        (story / "a.tiff").touch()
        (story / "b.tif").touch()
        (story / "c.webp").touch()

        discovery = StoryDiscovery(_make_io_config(story))
        result = discovery.discover()

        assert len(result) == 3

    def test_extension_matching_is_case_insensitive(self, tmp_path: Path) -> None:
        """Upper-case extensions like .JPG are included."""
        story = tmp_path / "story3"
        story.mkdir()
        (story / "A.JPG").touch()
        (story / "B.PNG").touch()

        discovery = StoryDiscovery(_make_io_config(story))
        result = discovery.discover()

        assert len(result) == 2
