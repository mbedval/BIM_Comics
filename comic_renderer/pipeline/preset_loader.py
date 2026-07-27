"""JSON preset loader.

Responsibility
--------------
Reads a ``<preset_name>.json`` file from the configured presets directory,
validates its structure, and returns a :class:`PresetConfig` dataclass.

The loader is **purely I/O + parsing** – it does not touch the filter
registry and does not validate whether the named filters actually exist.
That validation is the executor's job so the error message can include
useful context (which step failed, what filters are available, etc.).

Expected JSON schema
--------------------
.. code-block:: json

    {
        "name": "noir",
        "description": "High-contrast black and white comic style.",
        "version": "1.0",
        "pipeline": [
            {"filter": "grayscale",  "params": {}},
            {"filter": "clahe",      "params": {"clip_limit": 2.0}},
            {"filter": "posterize",  "params": {"levels": 4}}
        ]
    }

Required top-level keys: ``name``, ``description``, ``version``, ``pipeline``.
Each pipeline item requires a ``"filter"`` key; ``"params"`` is optional.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from comic_renderer.pipeline.models import FilterStep, PresetConfig

logger = logging.getLogger(__name__)

# Required top-level JSON keys.
_REQUIRED_KEYS: frozenset[str] = frozenset({"name", "description", "version", "pipeline"})


class PresetLoader:
    """Loads preset JSON files from the presets directory.

    Parameters
    ----------
    presets_dir:
        Directory that contains ``<preset_name>.json`` files.

    Raises
    ------
    FileNotFoundError
        When *presets_dir* does not exist.
    """

    def __init__(self, presets_dir: Path) -> None:
        if not presets_dir.exists():
            raise FileNotFoundError(
                f"Presets directory does not exist: {presets_dir}"
            )
        if not presets_dir.is_dir():
            raise FileNotFoundError(
                f"Presets path is not a directory: {presets_dir}"
            )
        self._presets_dir = presets_dir
        logger.debug("PresetLoader initialised with directory: %s", presets_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, preset_name: str) -> PresetConfig:
        """Load and parse the named preset.

        Parameters
        ----------
        preset_name:
            Bare name of the preset (e.g. ``"noir"``).  The loader
            appends ``.json`` automatically.

        Returns
        -------
        PresetConfig
            Fully parsed and validated preset.

        Raises
        ------
        FileNotFoundError
            When the JSON file is not found in the presets directory.
        ValueError
            When the JSON is malformed or missing required fields.
        """
        json_path = self._presets_dir / f"{preset_name}.json"

        if not json_path.exists():
            available = self.list_available()
            raise FileNotFoundError(
                f"Preset '{preset_name}' not found at: {json_path}\n"
                f"Available presets: {available}"
            )

        logger.debug("Loading preset from: %s", json_path)

        try:
            raw: dict = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Preset file is not valid JSON ({json_path}): {exc}"
            ) from exc

        return self._parse(raw, json_path)

    def list_available(self) -> list[str]:
        """Return the names of all preset JSON files in the presets directory.

        Returns
        -------
        list[str]
            Sorted list of preset names (without ``.json`` extension).
        """
        return sorted(p.stem for p in self._presets_dir.glob("*.json"))

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _parse(self, raw: dict, source_path: Path) -> PresetConfig:
        """Validate *raw* and convert to a :class:`PresetConfig`.

        Parameters
        ----------
        raw:
            Parsed JSON dictionary.
        source_path:
            Original file path (used in error messages only).

        Raises
        ------
        ValueError
            On any structural violation.
        """
        missing = _REQUIRED_KEYS - raw.keys()
        if missing:
            raise ValueError(
                f"Preset '{source_path.name}' is missing required keys: "
                f"{sorted(missing)}"
            )

        pipeline_raw = raw["pipeline"]
        if not isinstance(pipeline_raw, list):
            raise ValueError(
                f"Preset '{source_path.name}': 'pipeline' must be a JSON array."
            )

        steps: list[FilterStep] = []
        for idx, item in enumerate(pipeline_raw):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Preset '{source_path.name}': pipeline step {idx} must be "
                    "a JSON object."
                )
            if "filter" not in item:
                raise ValueError(
                    f"Preset '{source_path.name}': pipeline step {idx} is "
                    "missing the required 'filter' key."
                )
            filter_name: str = item["filter"]
            if not isinstance(filter_name, str) or not filter_name.strip():
                raise ValueError(
                    f"Preset '{source_path.name}': pipeline step {idx} has an "
                    "invalid 'filter' value (must be a non-empty string)."
                )
            params: dict = item.get("params", {})
            if not isinstance(params, dict):
                raise ValueError(
                    f"Preset '{source_path.name}': pipeline step {idx} 'params' "
                    "must be a JSON object."
                )
            steps.append(FilterStep(filter_name=filter_name.strip(), params=params))

        preset = PresetConfig(
            name=raw["name"],
            description=raw["description"],
            version=str(raw["version"]),
            steps=steps,
        )
        logger.debug(
            "Parsed preset '%s' with %d step(s).", preset.name, len(preset.steps)
        )
        return preset
