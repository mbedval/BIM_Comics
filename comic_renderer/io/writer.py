"""Image writer – saves NumPy arrays back to disk as PNG files.

Design decisions
----------------
* Output format is always **PNG** (lossless), regardless of input format.
  This avoids compounding lossy artefacts across repeated runs.
* The writer converts RGB → BGR before handing off to OpenCV.
* Parent directories are created automatically (``mkdir -p`` semantics).
* When ``overwrite=False`` and a file already exists the write is skipped
  and a WARNING is emitted rather than raising an exception, so a batch
  run can partially resume without crashing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from comic_renderer.config.settings import IOConfig

logger = logging.getLogger(__name__)

_PNG_EXTENSION = ".png"


class ImageWriter:
    """Writes processed image arrays to the output story directory.

    Parameters
    ----------
    io_config:
        Immutable I/O configuration injected at construction time.
    """

    def __init__(self, io_config: IOConfig) -> None:
        self._io_config = io_config

    def _resolve_output_path(self, source_path: Path) -> Path:
        """Return the output path for *source_path*, always using ``.png``."""
        return (
            self._io_config.output_story_dir()
            / (source_path.stem + _PNG_EXTENSION)
        )

    def write(self, image: np.ndarray, source_path: Path) -> Path:
        """Write *image* to disk and return the output path.

        Parameters
        ----------
        image:
            RGB ``uint8`` array with shape ``(H, W, 3)``.
        source_path:
            Path of the *source* image (used to derive the output filename).

        Returns
        -------
        Path
            The absolute path of the file that was written.

        Raises
        ------
        RuntimeError
            When OpenCV fails to encode/write the file.
        """
        output_path = self._resolve_output_path(source_path)

        if output_path.exists() and not self._io_config.overwrite:
            logger.warning(
                "Skipping '%s' – output already exists (use --overwrite to replace).",
                output_path.name,
            )
            return output_path

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # RGB → BGR for OpenCV
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        success = cv2.imwrite(str(output_path), bgr)
        if not success:
            raise RuntimeError(
                f"OpenCV failed to write image to: {output_path}"
            )

        logger.debug("Saved '%s' → '%s'", source_path.name, output_path)
        return output_path
