"""Unit tests for BaseFilter (filters.base).

Since BaseFilter is abstract, tests use lightweight anonymous concrete
subclasses defined locally inside the test methods.
"""

from __future__ import annotations

import numpy as np
import pytest

from comic_renderer.filters.base import BaseFilter


# ---------------------------------------------------------------------------
# Helpers – minimal concrete implementations
# ---------------------------------------------------------------------------


def _make_identity_filter_class(name: str = "identity_test") -> type[BaseFilter]:
    """Return a fresh concrete subclass that returns image unchanged."""

    class _IdentityFilter(BaseFilter):
        FILTER_NAME = name

        def apply(self, image: np.ndarray) -> np.ndarray:
            return image.copy()

    return _IdentityFilter


def _solid(r: int, g: int, b: int, h: int = 4, w: int = 4) -> np.ndarray:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = [r, g, b]
    return arr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBaseFilterAbstract:
    """BaseFilter cannot be instantiated directly."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            BaseFilter()  # type: ignore[abstract]

    def test_concrete_subclass_without_apply_cannot_instantiate(self) -> None:
        class _NoApply(BaseFilter):
            FILTER_NAME = "no_apply"
            # apply() not implemented

        with pytest.raises(TypeError):
            _NoApply()  # type: ignore[abstract]


class TestBaseFilterContract:
    """Concrete subclasses honour the BaseFilter contract."""

    def test_params_stored_on_construction(self) -> None:
        cls = _make_identity_filter_class()
        filt = cls(params={"alpha": 0.5, "beta": 10})
        assert filt.params == {"alpha": 0.5, "beta": 10}

    def test_none_params_stored_as_empty_dict(self) -> None:
        cls = _make_identity_filter_class()
        filt = cls(params=None)
        assert filt.params == {}

    def test_default_params_is_empty_dict(self) -> None:
        cls = _make_identity_filter_class()
        filt = cls()
        assert filt.params == {}

    def test_params_property_returns_copy(self) -> None:
        """Mutating the returned dict must not affect the stored params."""
        cls = _make_identity_filter_class()
        filt = cls(params={"key": "value"})
        returned = filt.params
        returned["key"] = "tampered"
        assert filt.params["key"] == "value"

    def test_apply_returns_ndarray(self) -> None:
        cls = _make_identity_filter_class()
        filt = cls()
        result = filt.apply(_solid(100, 150, 200))
        assert isinstance(result, np.ndarray)

    def test_apply_preserves_dtype(self) -> None:
        cls = _make_identity_filter_class()
        filt = cls()
        image = _solid(10, 20, 30)
        result = filt.apply(image)
        assert result.dtype == np.uint8

    def test_apply_preserves_shape(self) -> None:
        cls = _make_identity_filter_class()
        filt = cls()
        image = _solid(10, 20, 30, h=16, w=32)
        result = filt.apply(image)
        assert result.shape == (16, 32, 3)

    def test_filter_name_class_constant(self) -> None:
        cls = _make_identity_filter_class(name="my_filter")
        assert cls.FILTER_NAME == "my_filter"
