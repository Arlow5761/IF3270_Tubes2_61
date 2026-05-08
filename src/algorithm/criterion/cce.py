from __future__ import annotations
import numpy as np
from ..core.criterion import Criterion
from ..core.diffable import Diffable


class CCE(Criterion):
    """Categorical Cross-Entropy loss.

    Source: predictions of shape (N, C) — softmax output
    Target: one-hot encoded labels of shape (N, C)
    """

    def __init__(self, output: Diffable, target: np.ndarray):
        super().__init__(output, target)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        pred = next(iter(sources.values()))
        pred_c = np.clip(pred, 1e-15, 1.0)
        return np.array(-np.mean(np.sum(self._target * np.log(pred_c), axis=-1)))

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        pred_node, pred = next(iter(sources.items()))
        pred_c = np.clip(pred, 1e-15, 1.0)
        N = pred.shape[0]
        return {pred_node: -self._target / (pred_c * N)}
