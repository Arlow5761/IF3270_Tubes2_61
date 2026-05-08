from __future__ import annotations
import numpy as np
from ..core.criterion import Criterion
from ..core.diffable import Diffable


class RMSE(Criterion):
    """Root Mean Squared Error: sqrt(mean((pred - target)^2))."""

    def __init__(self, output: Diffable, target: np.ndarray):
        super().__init__(output, target)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        pred = next(iter(sources.values()))
        return np.array(np.sqrt(np.mean((pred - self._target) ** 2)))

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        pred_node, pred = next(iter(sources.items()))
        mse = np.mean((pred - self._target) ** 2)
        rmse_val = np.sqrt(mse) + 1e-15
        return {pred_node: (pred - self._target) / (pred.size * rmse_val)}
