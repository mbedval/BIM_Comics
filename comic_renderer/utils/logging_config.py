"""Logging configuration for the BIS Comic Renderer.

Call ``setup_logging`` exactly once at application start-up.
All other modules obtain a logger via ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import logging
import sys


_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(verbose: bool = False) -> None:
    """Configure the root logger with a single console handler.

    Parameters
    ----------
    verbose:
        When *True* the root level is set to ``DEBUG``; otherwise ``INFO``.
    """
    level = logging.DEBUG if verbose else logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(fmt=_CONSOLE_FORMAT, datefmt=_DATE_FORMAT)
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers when the function is called more than once
    # (e.g. during tests).
    if not root.handlers:
        root.addHandler(handler)
    else:
        root.handlers.clear()
        root.addHandler(handler)
