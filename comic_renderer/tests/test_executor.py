"""Integration tests for PipelineExecutor (pipeline.executor).

These tests exercise the full executor path: registry → filter creation →
apply() chain → result validation.  They do NOT depend on real preset JSON
files; presets are constructed in-memory.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from comic_renderer.filters.base import BaseFilter
from comic_renderer.filters.passthrough import PassThroughFilter
from comic_renderer.filters.registry import FilterRegistry
from comic_renderer.pipeline.executor import PipelineExecutor
from comic_renderer.pipeline.models import FilterStep, PresetConfig


# ---------------------------------------------------------------------------
# Test-only filter stubs
# ---------------------------------------------------------------------------


class _BrightnessFilter(BaseFilter):
    """Adds a fixed delta to every pixel channel (clamped to [0, 255])."""

    FILTER_NAME = "test_brightness"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._delta: int = int((params or {}).get("delta", 0))

    def apply(self, image: np.ndarray) -> np.ndarray:
        adjusted = image.astype(np.int32) + self._delta
        return np.clip(adjusted, 0, 255).astype(np.uint8)


class _ChannelNullFilter(BaseFilter):
    """Zeroes-out the R channel.  Used to verify pipeline ordering."""

    FILTER_NAME = "test_null_red"

    def apply(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        result[:, :, 0] = 0
        return result


class _BadDtypeFilter(BaseFilter):
    """Returns a float32 array – intentionally violates the contract."""

    FILTER_NAME = "test_bad_dtype"

    def apply(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32)  # Wrong dtype!


class _BadShapeFilter(BaseFilter):
    """Returns a 2-D array – intentionally violates the contract."""

    FILTER_NAME = "test_bad_shape"

    def apply(self, image: np.ndarray) -> np.ndarray:
        return image[:, :, 0]  # Drops colour channels!


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry(*filter_classes: type[BaseFilter]) -> FilterRegistry:
    registry = FilterRegistry()
    for cls in filter_classes:
        registry.register(cls)
    return registry


def _make_preset(
    *steps: tuple[str, dict],
    name: str = "test_preset",
) -> PresetConfig:
    return PresetConfig(
        name=name,
        description="Integration test preset.",
        version="1.0",
        steps=[FilterStep(filter_name=n, params=p) for n, p in steps],
    )


def _solid(r: int, g: int, b: int, h: int = 8, w: int = 8) -> np.ndarray:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = [r, g, b]
    return arr


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestPipelineExecutorHappyPath:

    def test_single_passthrough_returns_equal_array(self) -> None:
        registry = _make_registry(PassThroughFilter)
        executor = PipelineExecutor(registry=registry)
        preset = _make_preset(("passthrough", {}))
        image = _solid(100, 150, 200)

        result = executor.run(image, preset)

        np.testing.assert_array_equal(result, image)

    def test_result_is_a_copy_not_the_same_object(self) -> None:
        registry = _make_registry(PassThroughFilter)
        executor = PipelineExecutor(registry=registry)
        preset = _make_preset(("passthrough", {}))
        image = _solid(10, 20, 30)

        result = executor.run(image, preset)

        assert result is not image

    def test_empty_pipeline_returns_copy_of_input(self) -> None:
        registry = FilterRegistry()
        executor = PipelineExecutor(registry=registry)
        preset = _make_preset(name="empty")  # zero steps
        image = _solid(50, 100, 150)

        result = executor.run(image, preset)

        np.testing.assert_array_equal(result, image)
        assert result is not image

    def test_brightness_filter_increases_pixel_values(self) -> None:
        registry = _make_registry(_BrightnessFilter)
        executor = PipelineExecutor(registry=registry)
        preset = _make_preset(("test_brightness", {"delta": 50}))
        image = _solid(100, 100, 100)

        result = executor.run(image, preset)

        assert int(result[0, 0, 0]) == 150

    def test_pipeline_steps_applied_in_order(self) -> None:
        """Brightness (+50) then null-red → R=0, G/B=150."""
        registry = _make_registry(_BrightnessFilter, _ChannelNullFilter)
        executor = PipelineExecutor(registry=registry)
        preset = _make_preset(
            ("test_brightness", {"delta": 50}),
            ("test_null_red", {}),
        )
        image = _solid(100, 100, 100)

        result = executor.run(image, preset)

        assert result[0, 0, 0] == 0    # R nulled
        assert result[0, 0, 1] == 150  # G brightened
        assert result[0, 0, 2] == 150  # B brightened

    def test_brightness_clamped_at_255(self) -> None:
        registry = _make_registry(_BrightnessFilter)
        executor = PipelineExecutor(registry=registry)
        preset = _make_preset(("test_brightness", {"delta": 200}))
        image = _solid(200, 200, 200)

        result = executor.run(image, preset)

        assert np.all(result == 255)

    def test_output_dtype_is_uint8(self) -> None:
        registry = _make_registry(PassThroughFilter)
        executor = PipelineExecutor(registry=registry)
        preset = _make_preset(("passthrough", {}))

        result = executor.run(_solid(0, 0, 0), preset)

        assert result.dtype == np.uint8

    def test_output_shape_matches_input(self) -> None:
        registry = _make_registry(PassThroughFilter)
        executor = PipelineExecutor(registry=registry)
        preset = _make_preset(("passthrough", {}))
        image = np.zeros((120, 160, 3), dtype=np.uint8)

        result = executor.run(image, preset)

        assert result.shape == (120, 160, 3)

    def test_multiple_passthrough_steps_are_idempotent(self) -> None:
        registry = _make_registry(PassThroughFilter)
        executor = PipelineExecutor(registry=registry)
        preset = _make_preset(
            ("passthrough", {}),
            ("passthrough", {}),
            ("passthrough", {}),
        )
        image = _solid(77, 88, 99)

        result = executor.run(image, preset)

        np.testing.assert_array_equal(result, image)


class TestPipelineExecutorInputValidation:

    def test_wrong_dtype_raises_value_error(self) -> None:
        registry = FilterRegistry()
        executor = PipelineExecutor(registry=registry)
        preset = _make_preset(name="empty")
        bad_image = np.zeros((8, 8, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="uint8"):
            executor.run(bad_image, preset)

    def test_2d_image_raises_value_error(self) -> None:
        registry = FilterRegistry()
        executor = PipelineExecutor(registry=registry)
        preset = _make_preset(name="empty")
        bad_image = np.zeros((8, 8), dtype=np.uint8)

        with pytest.raises(ValueError, match=r"shape"):
            executor.run(bad_image, preset)

    def test_four_channel_image_raises_value_error(self) -> None:
        registry = FilterRegistry()
        executor = PipelineExecutor(registry=registry)
        preset = _make_preset(name="empty")
        bad_image = np.zeros((8, 8, 4), dtype=np.uint8)

        with pytest.raises(ValueError, match=r"shape"):
            executor.run(bad_image, preset)

    def test_non_array_input_raises_value_error(self) -> None:
        registry = FilterRegistry()
        executor = PipelineExecutor(registry=registry)
        preset = _make_preset(name="empty")

        with pytest.raises(ValueError):
            executor.run("not an array", preset)  # type: ignore[arg-type]


class TestPipelineExecutorErrorHandling:

    def test_unknown_filter_name_raises_key_error(self) -> None:
        registry = FilterRegistry()  # empty
        executor = PipelineExecutor(registry=registry)
        preset = _make_preset(("nonexistent_filter", {}))
        image = _solid(0, 0, 0)

        with pytest.raises(KeyError, match="nonexistent_filter"):
            executor.run(image, preset)

    def test_filter_returning_bad_dtype_raises_value_error(self) -> None:
        registry = _make_registry(_BadDtypeFilter)
        executor = PipelineExecutor(registry=registry)
        preset = _make_preset(("test_bad_dtype", {}))

        with pytest.raises(ValueError, match="dtype"):
            executor.run(_solid(10, 20, 30), preset)

    def test_filter_returning_bad_shape_raises_value_error(self) -> None:
        registry = _make_registry(_BadShapeFilter)
        executor = PipelineExecutor(registry=registry)
        preset = _make_preset(("test_bad_shape", {}))

        with pytest.raises(ValueError, match="shape"):
            executor.run(_solid(10, 20, 30), preset)
