"""Filter registry – the plugin loader for the pipeline system.

Responsibility
--------------
The ``FilterRegistry`` maps string names (as declared in JSON presets) to
concrete ``BaseFilter`` subclasses.  It is the **only** place that knows
about the mapping between name strings and Python types.

Design notes
------------
* Filters are registered explicitly via :meth:`register`, not discovered
  automatically.  This keeps the registry deterministic and easy to test.
* The registry holds **classes**, not instances.  A fresh instance is
  constructed for every pipeline step via :meth:`create`, which receives
  the per-step parameters from the preset JSON.
* Double-registration of the same name raises ``ValueError`` to prevent
  silent overrides during development.
"""

from __future__ import annotations

import logging
from typing import Any

from comic_renderer.filters.base import BaseFilter

logger = logging.getLogger(__name__)


class FilterRegistry:
    """Maintains a mapping of filter names → filter classes.

    Usage::

        registry = FilterRegistry()
        registry.register(GrayscaleFilter)
        registry.register(PassThroughFilter)

        # Later, inside the executor:
        filt = registry.create("grayscale", {"method": "luminance"})
        output = filt.apply(image)
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[BaseFilter]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, filter_class: type[BaseFilter]) -> None:
        """Register a filter class under its ``FILTER_NAME``.

        Parameters
        ----------
        filter_class:
            A concrete subclass of :class:`BaseFilter`.  Its ``FILTER_NAME``
            class attribute is used as the registry key.

        Raises
        ------
        TypeError
            When *filter_class* is not a subclass of ``BaseFilter``.
        ValueError
            When ``filter_class.FILTER_NAME`` is empty or already registered.
        """
        if not (isinstance(filter_class, type) and issubclass(filter_class, BaseFilter)):
            raise TypeError(
                f"Expected a BaseFilter subclass, got: {filter_class!r}"
            )
        name: str = filter_class.FILTER_NAME
        if not name:
            raise ValueError(
                f"Filter class {filter_class.__name__!r} has an empty FILTER_NAME."
            )
        if name in self._registry:
            raise ValueError(
                f"Filter '{name}' is already registered "
                f"(by {self._registry[name].__name__!r}). "
                "Each filter name must be unique."
            )
        self._registry[name] = filter_class
        logger.debug("Registered filter: '%s' → %s", name, filter_class.__name__)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def create(self, filter_name: str, params: dict[str, Any] | None = None) -> BaseFilter:
        """Instantiate a filter by name with the given parameters.

        Parameters
        ----------
        filter_name:
            The ``FILTER_NAME`` of the desired filter.
        params:
            Parameter dictionary forwarded to the filter constructor.
            Pass ``None`` or ``{}`` for filters that take no parameters.

        Returns
        -------
        BaseFilter
            A freshly constructed filter instance.

        Raises
        ------
        KeyError
            When *filter_name* is not registered.
        """
        if filter_name not in self._registry:
            available = ", ".join(repr(n) for n in sorted(self._registry))
            raise KeyError(
                f"Unknown filter: {filter_name!r}. "
                f"Available filters: [{available}]"
            )
        filter_class = self._registry[filter_name]
        instance = filter_class(params=params or {})
        logger.debug(
            "Created filter '%s' with params=%r", filter_name, params or {}
        )
        return instance

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def registered_names(self) -> list[str]:
        """Return a sorted list of all registered filter names."""
        return sorted(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """Return *True* if *name* is a registered filter."""
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:  # pragma: no cover
        return f"FilterRegistry(filters={self.registered_names()!r})"
