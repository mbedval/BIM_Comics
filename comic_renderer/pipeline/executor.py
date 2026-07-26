"""Pipeline executor – runs a sequence of filters on a single image.

Responsibility
--------------
The ``PipelineExecutor`` is the **only** component that orchestrates filter
execution.  No filter may call another filter directly.

Execution model
---------------
1. The preset's ``steps`` list is iterated in order.
2. For each step the executor asks the :class:`~comic_renderer.filters.registry.FilterRegistry`
   to instantiate the named filter with the step's ``params``.
3. The filter's ``apply()`` method is called on the current image array.
4. The returned array becomes the input to the next step.
5. Per-step timing is logged at DEBUG level.
6. The final array is returned to the caller.

Error handling
--------------
Unknown filter names raise :class:`KeyError` with a helpful message that
lists all available filters.  No partial results are written on failure.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from comic_renderer.filters.registry import FilterRegistry
from comic_renderer.pipeline.models import PresetConfig

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Executes a preset pipeline against a single image array.

    Parameters
    ----------
    registry:
        The filter registry used to resolve and instantiate each filter step.
        Injected at construction time so the executor is easily testable with
        a minimal registry.
    """

    def __init__(self, registry: FilterRegistry) -> None:
        self._registry = registry

    def run(self, image: np.ndarray, preset: PresetConfig) -> np.ndarray:
        """Apply every filter step in *preset* to *image* sequentially.

        Parameters
        ----------
        image:
            Input RGB ``uint8`` array with shape ``(H, W, 3)``.
        preset:
            Fully loaded preset configuration.

        Returns
        -------
        np.ndarray
            Processed RGB ``uint8`` array.

        Raises
        ------
        KeyError
            When a filter named in the preset is not registered.
        ValueError
            When the image array has an unexpected dtype or number of
            dimensions.
        """
        self._validate_image(image)

        if not preset.steps:
            logger.debug(
                "Preset '%s' has no steps – returning image unchanged.",
                preset.name,
            )
            return image.copy()

        logger.debug(
            "Executing preset '%s' (%d step(s)) on image %s.",
            preset.name,
            len(preset.steps),
            image.shape,
        )

        current: np.ndarray = image.copy()

        for step_idx, step in enumerate(preset.steps, start=1):
            step_start = time.perf_counter()

            # Resolve + instantiate the filter via the registry
            try:
                filt = self._registry.create(step.filter_name, step.params)
            except KeyError:
                raise KeyError(
                    f"Preset '{preset.name}', step {step_idx}: "
                    f"filter '{step.filter_name}' is not registered. "
                    f"Available filters: {self._registry.registered_names()}"
                )

            result = filt.apply(current)

            # Sanity-check that the filter returned a valid array.
            self._validate_output(result, step.filter_name, step_idx)

            elapsed_ms = (time.perf_counter() - step_start) * 1000
            logger.debug(
                "  Step %d/%d '%s' → %.1f ms",
                step_idx,
                len(preset.steps),
                step.filter_name,
                elapsed_ms,
            )
            current = result

        return current

    # ------------------------------------------------------------------
    # Internal validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """Raise ``ValueError`` if *image* is not a valid input array."""
        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"Expected np.ndarray input, got {type(image).__name__!r}."
            )
        if image.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 image, got dtype={image.dtype}."
            )
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected shape (H, W, 3), got {image.shape}."
            )

    @staticmethod
    def _validate_output(
        result: np.ndarray, filter_name: str, step_idx: int
    ) -> None:
        """Raise ``ValueError`` if a filter returned an invalid array."""
        if not isinstance(result, np.ndarray):
            raise ValueError(
                f"Filter '{filter_name}' (step {step_idx}) returned "
                f"{type(result).__name__!r} instead of np.ndarray."
            )
        if result.dtype != np.uint8:
            raise ValueError(
                f"Filter '{filter_name}' (step {step_idx}) returned "
                f"dtype={result.dtype}; expected uint8."
            )
        if result.ndim != 3 or result.shape[2] != 3:
            raise ValueError(
                f"Filter '{filter_name}' (step {step_idx}) returned "
                f"shape {result.shape}; expected (H, W, 3)."
            )
