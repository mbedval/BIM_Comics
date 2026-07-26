"""Story folder discovery utilities.

Responsible for scanning a story directory and returning an ordered list of
image paths that the pipeline should process.  Discovery is pure I/O – it does
not open or decode image data.
"""

from __future__ import annotations

import logging
from pathlib import Path

from comic_renderer.config.settings import IOConfig

logger = logging.getLogger(__name__)


class StoryDiscovery:
    """Discovers image files in a story directory.

    Parameters
    ----------
    io_config:
        Immutable I/O configuration injected at construction time.
    """

    def __init__(self, io_config: IOConfig) -> None:
        self._io_config = io_config

    def discover(self) -> list[Path]:
        """Return a sorted list of supported image paths in the story directory.

        The list is sorted lexicographically by filename so that ``001.jpg``
        comes before ``002.jpg``.

        Returns
        -------
        list[Path]
            Sorted image paths.  Empty list if no matching files are found.

        Raises
        ------
        FileNotFoundError
            When ``story_path`` does not exist or is not a directory.
        """
        story_path = self._io_config.story_path

        if not story_path.exists():
            raise FileNotFoundError(
                f"Story path does not exist: {story_path}"
            )
        if not story_path.is_dir():
            raise FileNotFoundError(
                f"Story path is not a directory: {story_path}"
            )

        supported = self._io_config.supported_extensions
        images: list[Path] = [
            p
            for p in story_path.iterdir()
            if p.is_file() and p.suffix.lower() in supported
        ]
        images.sort(key=lambda p: p.name)

        logger.debug(
            "Discovered %d image(s) in '%s'", len(images), story_path
        )
        return images
