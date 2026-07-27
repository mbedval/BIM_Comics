"""Unit tests for AnimeFilter (filters.anime)."""

from __future__ import annotations

import numpy as np
import pytest

from comic_renderer.filters.anime import AnimeFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid(value: int = 128, h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _apply(image: np.ndarray, **params) -> np.ndarray:
    return AnimeFilter(params=params).apply(image)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestAnimeFilterValidation:

    def test_invalid_max_dim_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="max_dim"):
            AnimeFilter({"max_dim": 16})

    def test_valid_max_dim_does_not_raise(self) -> None:
        filt = AnimeFilter({"max_dim": 64})
        assert filt.params["max_dim"] == 64


# ---------------------------------------------------------------------------
# Contract & Functional tests
# ---------------------------------------------------------------------------


class TestAnimeFilterFunctional:

    def test_output_is_uint8(self) -> None:
        result = _apply(_solid())
        assert result.dtype == np.uint8

    def test_output_has_three_channels(self) -> None:
        result = _apply(_solid())
        assert result.ndim == 3 and result.shape[2] == 3

    def test_shape_preserved(self) -> None:
        image = np.zeros((32, 64, 3), dtype=np.uint8)
        result = _apply(image)
        assert result.shape == (32, 64, 3)

    def test_output_values_in_uint8_range(self) -> None:
        result = _apply(_solid(255))
        assert result.min() >= 0 and result.max() <= 255

    def test_serialization_getstate_excludes_session(self) -> None:
        filt = AnimeFilter()
        # Trigger session initialization
        _ = filt.apply(_solid(128, 32, 32))
        assert filt._session is not None

        # Get state
        state = filt.__getstate__()
        assert state["_session"] is None
