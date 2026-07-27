"""Unit tests for PresetLoader (pipeline.preset_loader) and the
FilterStep / PresetConfig dataclasses (pipeline.models).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from comic_renderer.pipeline.models import FilterStep, PresetConfig
from comic_renderer.pipeline.preset_loader import PresetLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_preset(directory: Path, name: str, data: dict) -> Path:
    """Write *data* as JSON to ``<directory>/<name>.json``."""
    path = directory / f"{name}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _minimal_preset(name: str = "test_preset") -> dict:
    return {
        "name": name,
        "description": "A test preset.",
        "version": "1.0",
        "pipeline": [
            {"filter": "passthrough", "params": {}},
        ],
    }


# ---------------------------------------------------------------------------
# FilterStep dataclass tests
# ---------------------------------------------------------------------------


class TestFilterStep:
    def test_requires_filter_name(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            FilterStep(filter_name="")

    def test_default_params_is_empty_dict(self) -> None:
        step = FilterStep(filter_name="grayscale")
        assert step.params == {}

    def test_stores_params(self) -> None:
        step = FilterStep(filter_name="clahe", params={"clip_limit": 2.0})
        assert step.params["clip_limit"] == 2.0


# ---------------------------------------------------------------------------
# PresetConfig dataclass tests
# ---------------------------------------------------------------------------


class TestPresetConfig:
    def test_stores_all_fields(self) -> None:
        steps = [FilterStep("passthrough")]
        cfg = PresetConfig(
            name="noir",
            description="Test",
            version="1.0",
            steps=steps,
        )
        assert cfg.name == "noir"
        assert cfg.version == "1.0"
        assert len(cfg.steps) == 1

    def test_empty_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="name"):
            PresetConfig(name="", description="d", version="1.0", steps=[])


# ---------------------------------------------------------------------------
# PresetLoader tests
# ---------------------------------------------------------------------------


class TestPresetLoaderInit:
    def test_missing_directory_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PresetLoader(presets_dir=tmp_path / "nonexistent")

    def test_file_path_raises_file_not_found(self, tmp_path: Path) -> None:
        f = tmp_path / "not_a_dir.json"
        f.touch()
        with pytest.raises(FileNotFoundError):
            PresetLoader(presets_dir=f)

    def test_valid_directory_initialises_successfully(self, tmp_path: Path) -> None:
        loader = PresetLoader(presets_dir=tmp_path)
        assert loader is not None


class TestPresetLoaderLoad:
    def test_loads_valid_preset(self, tmp_path: Path) -> None:
        _write_preset(tmp_path, "noir", _minimal_preset("noir"))
        loader = PresetLoader(presets_dir=tmp_path)
        cfg = loader.load("noir")
        assert cfg.name == "noir"
        assert cfg.version == "1.0"

    def test_pipeline_steps_are_parsed(self, tmp_path: Path) -> None:
        data = _minimal_preset()
        data["pipeline"] = [
            {"filter": "grayscale", "params": {}},
            {"filter": "clahe", "params": {"clip_limit": 2.5}},
        ]
        _write_preset(tmp_path, "test_preset", data)
        loader = PresetLoader(presets_dir=tmp_path)
        cfg = loader.load("test_preset")

        assert len(cfg.steps) == 2
        assert cfg.steps[0].filter_name == "grayscale"
        assert cfg.steps[1].filter_name == "clahe"
        assert cfg.steps[1].params["clip_limit"] == 2.5

    def test_missing_preset_raises_file_not_found(self, tmp_path: Path) -> None:
        loader = PresetLoader(presets_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="ghost"):
            loader.load("ghost")

    def test_invalid_json_raises_value_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "broken.json"
        bad.write_text("{not valid json", encoding="utf-8")
        loader = PresetLoader(presets_dir=tmp_path)
        with pytest.raises(ValueError, match="not valid JSON"):
            loader.load("broken")

    def test_missing_required_key_raises_value_error(self, tmp_path: Path) -> None:
        data = _minimal_preset()
        del data["description"]  # Remove a required key
        _write_preset(tmp_path, "bad_preset", data)
        loader = PresetLoader(presets_dir=tmp_path)
        with pytest.raises(ValueError, match="missing required keys"):
            loader.load("bad_preset")

    def test_pipeline_not_a_list_raises_value_error(self, tmp_path: Path) -> None:
        data = _minimal_preset()
        data["pipeline"] = "not a list"
        _write_preset(tmp_path, "bad_pipeline", data)
        loader = PresetLoader(presets_dir=tmp_path)
        with pytest.raises(ValueError, match="pipeline.*array"):
            loader.load("bad_pipeline")

    def test_step_missing_filter_key_raises_value_error(self, tmp_path: Path) -> None:
        data = _minimal_preset()
        data["pipeline"] = [{"params": {}}]  # No "filter" key
        _write_preset(tmp_path, "no_filter_key", data)
        loader = PresetLoader(presets_dir=tmp_path)
        with pytest.raises(ValueError, match="missing the required 'filter' key"):
            loader.load("no_filter_key")

    def test_step_params_optional(self, tmp_path: Path) -> None:
        """'params' key is optional in a pipeline step."""
        data = _minimal_preset()
        data["pipeline"] = [{"filter": "grayscale"}]  # No "params"
        _write_preset(tmp_path, "no_params", data)
        loader = PresetLoader(presets_dir=tmp_path)
        cfg = loader.load("no_params")
        assert cfg.steps[0].params == {}

    def test_empty_pipeline_is_valid(self, tmp_path: Path) -> None:
        """A preset with zero steps is structurally valid."""
        data = _minimal_preset()
        data["pipeline"] = []
        _write_preset(tmp_path, "empty_pipeline", data)
        loader = PresetLoader(presets_dir=tmp_path)
        cfg = loader.load("empty_pipeline")
        assert cfg.steps == []


class TestPresetLoaderListAvailable:
    def test_lists_json_files_as_preset_names(self, tmp_path: Path) -> None:
        for name in ("alpha", "beta", "gamma"):
            _write_preset(tmp_path, name, _minimal_preset(name))
        loader = PresetLoader(presets_dir=tmp_path)
        assert loader.list_available() == ["alpha", "beta", "gamma"]

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        loader = PresetLoader(presets_dir=tmp_path)
        assert loader.list_available() == []

    def test_non_json_files_are_excluded(self, tmp_path: Path) -> None:
        _write_preset(tmp_path, "good", _minimal_preset("good"))
        (tmp_path / "readme.txt").write_text("ignore me")
        loader = PresetLoader(presets_dir=tmp_path)
        assert loader.list_available() == ["good"]

    def test_result_is_sorted(self, tmp_path: Path) -> None:
        for name in ("zebra", "apple", "mango"):
            _write_preset(tmp_path, name, _minimal_preset(name))
        loader = PresetLoader(presets_dir=tmp_path)
        names = loader.list_available()
        assert names == sorted(names)
