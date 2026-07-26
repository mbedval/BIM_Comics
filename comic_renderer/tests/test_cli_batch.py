"""Unit and integration smoke tests for CLI batch processing with multiprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from comic_renderer.bis_comic_main import build_parser, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tiny_png(path: Path, h: int = 4, w: int = 4) -> None:
    """Write a solid-red PNG to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.full((h, w, 3), [200, 50, 50], dtype=np.uint8), mode="RGB"
    ).save(str(path))


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestCLIJobsParser:

    def test_default_jobs_is_one(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--storypath", "/tmp/x"])
        assert args.jobs == 1

    def test_custom_jobs_parsed(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--storypath", "/tmp/x", "--jobs", "4"])
        assert args.jobs == 4

        args_short = parser.parse_args(["--storypath", "/tmp/x", "-j", "0"])
        assert args_short.jobs == 0

    def test_negative_jobs_validation_raises(self) -> None:
        """A negative number of jobs should cause the CLI to exit with code 2."""
        with pytest.raises(SystemExit) as excinfo:
            main(["--storypath", "/tmp/x", "--jobs", "-2"])
        assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# End-to-end parallel integration tests
# ---------------------------------------------------------------------------


class TestMainParallelEndToEnd:

    def test_parallel_batch_execution_exits_zero(self, tmp_path: Path) -> None:
        """Run parallel execution with 2 jobs on 2 images."""
        story = tmp_path / "story1"
        _write_tiny_png(story / "001.png", h=8, w=8)
        _write_tiny_png(story / "002.png", h=8, w=8)
        output = tmp_path / "output"

        # Run with jobs=2
        exit_code = main([
            "--storypath", str(story),
            "--preset", "graphic_novel",
            "--output", str(output),
            "--jobs", "2",
            "--overwrite"
        ])
        assert exit_code == 0

        # Output folder output/story1 should exist and contain outputs
        out_dir = output / "story1"
        assert out_dir.exists()
        assert (out_dir / "001.png").exists()
        assert (out_dir / "002.png").exists()

    def test_parallel_batch_execution_auto_jobs(self, tmp_path: Path) -> None:
        """Run parallel execution with auto core detection (jobs=0)."""
        story = tmp_path / "story2"
        _write_tiny_png(story / "abc.jpg", h=8, w=8)
        output = tmp_path / "output"

        exit_code = main([
            "--storypath", str(story),
            "--preset", "noir",
            "--output", str(output),
            "--jobs", "0",
            "--overwrite"
        ])
        assert exit_code == 0
        assert (output / "story2" / "abc.png").exists()
