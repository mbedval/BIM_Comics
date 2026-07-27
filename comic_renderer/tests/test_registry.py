"""Unit tests for FilterRegistry (filters.registry).

Tests use locally-defined minimal filter stubs to keep each test self-contained.
"""

from __future__ import annotations

import numpy as np
import pytest

from comic_renderer.filters.base import BaseFilter
from comic_renderer.filters.registry import FilterRegistry
from comic_renderer.filters.passthrough import PassThroughFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_filter_class(name: str) -> type[BaseFilter]:
    """Return a uniquely-named filter class for use in registry tests."""

    class _StubFilter(BaseFilter):
        FILTER_NAME = name

        def apply(self, image: np.ndarray) -> np.ndarray:
            return image.copy()

    # Give the class a distinct __name__ for clearer error messages.
    _StubFilter.__name__ = f"Stub_{name.title()}Filter"
    return _StubFilter


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFilterRegistryRegistration:
    """Tests covering the register() method."""

    def test_register_adds_filter(self) -> None:
        registry = FilterRegistry()
        cls = _make_filter_class("alpha")
        registry.register(cls)
        assert registry.is_registered("alpha")

    def test_register_multiple_filters(self) -> None:
        registry = FilterRegistry()
        registry.register(_make_filter_class("filter_a"))
        registry.register(_make_filter_class("filter_b"))
        registry.register(_make_filter_class("filter_c"))
        assert len(registry) == 3

    def test_double_registration_raises_value_error(self) -> None:
        registry = FilterRegistry()
        cls = _make_filter_class("duplicate")
        registry.register(cls)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(cls)

    def test_non_base_filter_subclass_raises_type_error(self) -> None:
        registry = FilterRegistry()
        with pytest.raises(TypeError):
            registry.register(int)  # type: ignore[arg-type]

    def test_empty_filter_name_raises_value_error(self) -> None:
        registry = FilterRegistry()

        class _BadFilter(BaseFilter):
            FILTER_NAME = ""  # violates the contract

            def apply(self, image: np.ndarray) -> np.ndarray:
                return image.copy()

        with pytest.raises(ValueError, match="empty FILTER_NAME"):
            registry.register(_BadFilter)

    def test_passthrough_filter_can_be_registered(self) -> None:
        registry = FilterRegistry()
        registry.register(PassThroughFilter)
        assert registry.is_registered("passthrough")


class TestFilterRegistryCreate:
    """Tests covering the create() method."""

    def test_create_returns_instance_of_correct_class(self) -> None:
        registry = FilterRegistry()
        cls = _make_filter_class("my_filter")
        registry.register(cls)
        instance = registry.create("my_filter")
        assert isinstance(instance, cls)

    def test_create_passes_params_to_instance(self) -> None:
        registry = FilterRegistry()
        registry.register(_make_filter_class("parameterised"))
        instance = registry.create("parameterised", {"alpha": 42})
        assert instance.params == {"alpha": 42}

    def test_create_with_no_params_produces_empty_dict(self) -> None:
        registry = FilterRegistry()
        registry.register(_make_filter_class("noparam"))
        instance = registry.create("noparam")
        assert instance.params == {}

    def test_create_unknown_filter_raises_key_error(self) -> None:
        registry = FilterRegistry()
        with pytest.raises(KeyError, match="ghost"):
            registry.create("ghost")

    def test_create_returns_fresh_instance_each_call(self) -> None:
        """create() must not return the same object twice (no singletons)."""
        registry = FilterRegistry()
        registry.register(_make_filter_class("fresh"))
        a = registry.create("fresh")
        b = registry.create("fresh")
        assert a is not b


class TestFilterRegistryIntrospection:
    """Tests covering registered_names() and is_registered()."""

    def test_registered_names_returns_sorted_list(self) -> None:
        registry = FilterRegistry()
        registry.register(_make_filter_class("zebra"))
        registry.register(_make_filter_class("apple"))
        registry.register(_make_filter_class("mango"))
        names = registry.registered_names()
        assert names == sorted(names)

    def test_registered_names_empty_registry(self) -> None:
        assert FilterRegistry().registered_names() == []

    def test_is_registered_returns_false_for_unknown(self) -> None:
        registry = FilterRegistry()
        assert registry.is_registered("unknown") is False

    def test_len_reflects_registration_count(self) -> None:
        registry = FilterRegistry()
        assert len(registry) == 0
        registry.register(_make_filter_class("one"))
        assert len(registry) == 1
        registry.register(_make_filter_class("two"))
        assert len(registry) == 2

    def test_register_alias(self) -> None:
        registry = FilterRegistry()
        cls = _make_filter_class("original")
        registry.register(cls)
        registry.register_alias("alias_name", cls)
        assert registry.is_registered("alias_name")
        instance = registry.create("alias_name")
        assert isinstance(instance, cls)

    def test_all_presets_are_valid_in_registry(self) -> None:
        from comic_renderer.bis_comic_main import _build_registry, _PRESETS_DIR
        from comic_renderer.pipeline.preset_loader import PresetLoader
        registry = _build_registry()
        loader = PresetLoader(presets_dir=_PRESETS_DIR)
        for name in loader.list_available():
            preset = loader.load(name)
            # Ensure every step in the preset can be successfully instantiated by the registry
            for step in preset.steps:
                assert registry.is_registered(step.filter_name), f"Filter '{step.filter_name}' in preset '{name}' is not registered."
                # Instantiate to verify validation passes
                registry.create(step.filter_name, step.params)

