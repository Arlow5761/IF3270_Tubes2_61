from __future__ import annotations
import numpy as np
from ..core.criterion import Criterion
from ..core.diffable import Diffable


class MSE(Criterion):
    """Mean Squared Error: mean((pred - target)^2)."""

    def __init__(self, output: Diffable, target: np.ndarray):
        super().__init__(output, target)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        pred = next(iter(sources.values()))
        return np.array(np.mean((pred - self._target) ** 2))

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        pred_node, pred = next(iter(sources.items()))
        return {pred_node: 2.0 * (pred - self._target) / pred.size}
