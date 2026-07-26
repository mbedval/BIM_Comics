"""Dataclasses representing a loaded preset configuration.

These are pure data containers with no behaviour.  They are produced by
:class:`~comic_renderer.pipeline.preset_loader.PresetLoader` and consumed by
:class:`~comic_renderer.pipeline.executor.PipelineExecutor`.

Keeping them in a dedicated module avoids circular imports between the
loader and executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FilterStep:
    """One step in a preset pipeline.

    Attributes
    ----------
    filter_name:
        The registry key for the filter (matches ``FILTER_NAME`` on the
        concrete ``BaseFilter`` subclass).
    params:
        Keyword arguments forwarded verbatim to the filter constructor.
    """

    filter_name: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.filter_name:
            raise ValueError("FilterStep.filter_name must not be empty.")


@dataclass
class PresetConfig:
    """Complete configuration for a named rendering preset.

    Attributes
    ----------
    name:
        Canonical preset name (e.g. ``"noir"``).
    description:
        Human-readable description shown in ``--list-presets``.
    version:
        Schema version string (e.g. ``"1.0"``).  Reserved for future
        backward-compatibility handling.
    steps:
        Ordered list of filter steps that the executor runs in sequence.
    """

    name: str
    description: str
    version: str
    steps: list[FilterStep]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("PresetConfig.name must not be empty.")
