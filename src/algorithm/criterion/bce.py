from __future__ import annotations
import numpy as np
from ..core.criterion import Criterion
from ..core.diffable import Diffable


class BCE(Criterion):
    """Binary Cross-Entropy loss.

    Source: predictions of shape (N,) or (N, 1) — sigmoid output in [0, 1]
    Target: binary labels (0 or 1) of the same shape
    """

    def __init__(self, output: Diffable, target: np.ndarray):
        super().__init__(output, target)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        pred = next(iter(sources.values()))
        pred_c = np.clip(pred, 1e-15, 1.0 - 1e-15)
        return np.array(-np.mean(
            self._target * np.log(pred_c) + (1.0 - self._target) * np.log(1.0 - pred_c)))

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        pred_node, pred = next(iter(sources.items()))
        pred_c = np.clip(pred, 1e-15, 1.0 - 1e-15)
        N = pred.size
        grad = (-self._target / pred_c + (1.0 - self._target) / (1.0 - pred_c)) / N
        return {pred_node: grad}
