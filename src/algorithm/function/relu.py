from __future__ import annotations
import numpy as np
from ..core.diffable import Diffable


class ReLU(Diffable):
    """Element-wise Rectified Linear Unit activation."""

    def __init__(self, x: Diffable):
        super().__init__(x)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        x = next(iter(sources.values()))
        return np.maximum(0.0, x)

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        x_node, x = next(iter(sources.items()))
        return {x_node: (x > 0).astype(np.float64)}
