from __future__ import annotations
import numpy as np
from ..core.diffable import Diffable


class Sigmoid(Diffable):
    """Element-wise sigmoid activation: 1 / (1 + exp(-x))."""

    def __init__(self, x: Diffable):
        super().__init__(x)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        x = next(iter(sources.values()))
        return np.where(x >= 0,
                        1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        x_node = next(iter(sources.keys()))
        return {x_node: value * (1.0 - value)}
