from __future__ import annotations
import numpy as np
from ..core.diffable import Diffable


class Tanh(Diffable):
    """Element-wise hyperbolic tangent activation."""

    def __init__(self, x: Diffable):
        super().__init__(x)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        x = next(iter(sources.values()))
        return np.tanh(x)

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        x_node = next(iter(sources.keys()))
        return {x_node: 1.0 - value ** 2}
