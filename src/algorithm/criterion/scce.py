from __future__ import annotations
import numpy as np
from ..core.criterion import Criterion
from ..core.diffable import Diffable


class SCCE(Criterion):
    """Sparse Categorical Cross-Entropy loss.

    Expects softmax probabilities as input (not raw logits).

    Source: predictions of shape (N, num_classes) — softmax output
    Target: integer class indices of shape (N,)
    Output: scalar loss value
    """

    def __init__(self, output: Diffable, target: np.ndarray):
        super().__init__(output, target)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        pred = next(iter(sources.values()))   # (N, C)
        N = pred.shape[0]
        labels = self._target.astype(int)
        pred_clipped = np.clip(pred, 1e-15, 1.0)
        return np.array(-np.mean(np.log(pred_clipped[np.arange(N), labels])))

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        pred_node, pred = next(iter(sources.items()))
        N = pred.shape[0]
        labels = self._target.astype(int)
        pred_clipped = np.clip(pred, 1e-15, 1.0)
        grad = np.zeros_like(pred)
        grad[np.arange(N), labels] = -1.0 / (N * pred_clipped[np.arange(N), labels])
        return {pred_node: grad}
