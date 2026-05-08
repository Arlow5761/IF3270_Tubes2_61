from __future__ import annotations
import numpy as np
from .diffable import Diffable


class Parameter(Diffable):
    """Leaf node for trainable weights/biases. Persists value across forward passes."""

    _warn_on_no_sources = False

    def __init__(self, value: np.ndarray):
        super().__init__()
        self._value = value

    def clear_values(self):
        # Do not clear _value — weights must persist between forward passes.
        # Only propagate clearing to downstream nodes.
        self._state.clear()
        for source in self._sources:
            self._sources[source] = None
        for sink in self._sinks:
            sink.clear_values()

    def _calculate_value(self, sources: dict) -> np.ndarray:
        return self._value

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        return {}


class Input(Diffable):
    """Leaf node for external input data. Value is set before each forward pass."""

    _warn_on_no_sources = False

    def __init__(self):
        super().__init__()

    def _calculate_value(self, sources: dict) -> np.ndarray:
        return self._value

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        return {}
