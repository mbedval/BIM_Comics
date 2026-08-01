"""Shared configuration dataclasses for the BIS Comic Renderer.

These dataclasses are the single source of truth for all runtime options.
They are populated by the CLI argument parser and passed by dependency
injection into every component that needs them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class IOConfig:
    """Paths and I/O policy for a single run."""

    story_path: Path
    """Source directory that contains the raw input images."""

    output_root: Path
    """Root directory where processed stories are saved."""

    overwrite: bool = False
    """When *True* existing output files are silently replaced."""

    supported_extensions: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        )
    )
    """Image file extensions that the loader accepts (lowercase, with dot)."""

    def output_story_dir(self) -> Path:
        """Return the output directory for the current story.

        The story name is taken from the *last component* of ``story_path``
        so that ``images/input/story1`` maps to ``<output_root>/story1``.
        """
        return self.output_root / self.story_path.name


@dataclass(frozen=True)
class LoggingConfig:
    """Logging verbosity configuration."""

    verbose: bool = False
    """When *True*, the root logger is set to DEBUG; otherwise INFO."""


@dataclass(frozen=True)
class RunConfig:
    """Top-level runtime configuration assembled from CLI arguments."""

    io: IOConfig
    logging: LoggingConfig
    preset_name: Optional[str]
    """Name of the JSON preset to apply (e.g. ``"graphic_novel"``). When None, all presets are executed."""

    jobs: int = 1
    """Number of parallel worker processes. 0 means use all available CPUs."""

    remove_bg: bool = False
    """When *True*, the background is removed using the withoutbg package before rendering."""

    bg_color: str = "white"
    """Background color to fill when background is removed (e.g. 'white', 'black')."""
