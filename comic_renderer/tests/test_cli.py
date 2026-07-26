"""Smoke tests for the CLI (bis_comic_main.py).

These tests invoke main() directly (not via subprocess) so they run quickly
and give precise stack traces.  Each test covers one observable CLI behaviour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from comic_renderer.bis_comic_main import build_parser, list_presets, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tiny_png(path: Path) -> None:
    """Write a 4×4 solid-red PNG to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.full((4, 4, 3), [200, 50, 50], dtype=np.uint8), mode="RGB"
    ).save(str(path))


# ---------------------------------------------------------------------------
# Parser smoke tests
# ---------------------------------------------------------------------------


class TestBuildParser:
    """Verify that the argument parser is correctly constructed."""

    def test_returns_parser_instance(self) -> None:
        import argparse
        assert isinstance(build_parser(), argparse.ArgumentParser)

    def test_default_preset_is_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--storypath", "/tmp/x"])
        assert args.preset is None

    def test_default_overwrite_is_false(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--storypath", "/tmp/x"])
        assert args.overwrite is False

    def test_verbose_flag_sets_attribute(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--storypath", "/tmp/x", "--verbose"])
        assert args.verbose is True

    def test_list_presets_flag_sets_attribute(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--list-presets"])
        assert args.list_presets is True


# ---------------------------------------------------------------------------
# --list-presets smoke test
# ---------------------------------------------------------------------------


class TestListPresets:
    """--list-presets must print names and exit 0."""

    def test_list_presets_exits_zero(self, capsys: pytest.CaptureFixture) -> None:
        exit_code = main(["--list-presets"])
        assert exit_code == 0

    def test_list_presets_prints_known_names(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        main(["--list-presets"])
        output = capsys.readouterr().out
        for name in ("noir", "pencil_comic"):
            assert name in output


# ---------------------------------------------------------------------------
# End-to-end smoke tests
# ---------------------------------------------------------------------------


class TestMainEndToEnd:
    """Integration smoke tests – run main() with real tmp_path files."""

    def test_processes_single_image_and_exits_zero(self, tmp_path: Path) -> None:
        story = tmp_path / "story1"
        _write_tiny_png(story / "001.png")
        output = tmp_path / "output"

        exit_code = main([
            "--storypath", str(story),
            "--preset", "noir",
            "--output", str(output),
        ])

        assert exit_code == 0
        assert (output / "story1" / "001.png").exists()

    def test_processes_multiple_images(self, tmp_path: Path) -> None:
        story = tmp_path / "story2"
        for i in range(1, 4):
            _write_tiny_png(story / f"{i:03d}.png")
        output = tmp_path / "output"

        exit_code = main([
            "--storypath", str(story),
            "--preset", "noir",
            "--output", str(output),
        ])

        assert exit_code == 0
        for i in range(1, 4):
            assert (output / "story2" / f"{i:03d}.png").exists()

    def test_missing_storypath_exits_nonzero(self, tmp_path: Path) -> None:
        """A non-existent story path should return exit code 1."""
        exit_code = main([
            "--storypath", str(tmp_path / "ghost_story"),
            "--preset", "noir",
            "--output", str(tmp_path / "output"),
        ])
        assert exit_code == 1

    def test_unknown_preset_triggers_parser_error(self) -> None:
        """An unknown preset name must trigger a SystemExit."""
        with pytest.raises(SystemExit):
            main(["--storypath", "/tmp/x", "--preset", "does_not_exist"])

    def test_overwrite_flag_replaces_existing_output(self, tmp_path: Path) -> None:
        story = tmp_path / "story3"
        _write_tiny_png(story / "001.png")
        output = tmp_path / "output"

        # First run
        main([
            "--storypath", str(story),
            "--output", str(output),
            "--preset", "noir",
        ])
        first_mtime = (output / "story3" / "001.png").stat().st_mtime

        # Second run without --overwrite (should skip)
        main([
            "--storypath", str(story),
            "--output", str(output),
            "--preset", "noir",
        ])
        second_mtime = (output / "story3" / "001.png").stat().st_mtime
        assert first_mtime == second_mtime  # Not touched

        # Third run with --overwrite
        import time; time.sleep(0.05)  # Ensure mtime resolution
        main([
            "--storypath", str(story),
            "--output", str(output),
            "--preset", "noir",
            "--overwrite",
        ])
        third_mtime = (output / "story3" / "001.png").stat().st_mtime
        assert third_mtime > first_mtime  # Was replaced

    def test_empty_story_dir_exits_zero_without_crash(self, tmp_path: Path) -> None:
        """An empty story directory should log a warning and exit 0."""
        story = tmp_path / "empty"
        story.mkdir()
        output = tmp_path / "output"

        exit_code = main([
            "--storypath", str(story),
            "--preset", "noir",
            "--output", str(output),
        ])
        assert exit_code == 0

    def test_no_preset_runs_all_presets_with_suffix(self, tmp_path: Path) -> None:
        """When no --preset is specified, all presets should run with a filename suffix."""
        story = tmp_path / "story_all"
        _write_tiny_png(story / "001.png")
        output = tmp_path / "output_all"

        exit_code = main([
            "--storypath", str(story),
            "--output", str(output),
            "--jobs", "1",
        ])
        assert exit_code == 0

        # Discover available presets directly
        from comic_renderer.bis_comic_main import _PRESETS_DIR
        from comic_renderer.pipeline.preset_loader import PresetLoader
        available = PresetLoader(presets_dir=_PRESETS_DIR).list_available()

        # Check that output files exist for each preset with the suffix
        for preset_name in available:
            expected_file = output / "story_all" / f"001_{preset_name}.png"
            assert expected_file.exists()

