from __future__ import annotations
import numpy as np
from ..core.criterion import Criterion
from ..core.diffable import Diffable


class CosineSimilarityLoss(Criterion):
    """Cosine similarity loss: 1 - cosine_similarity(pred, target).

    Source: predictions of shape (N, D)
    Target: target vectors of shape (N, D)
    """

    def __init__(self, output: Diffable, target: np.ndarray):
        super().__init__(output, target)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        pred = next(iter(sources.values()))
        eps = 1e-15
        pred_norm   = np.linalg.norm(pred,          axis=-1, keepdims=True) + eps
        target_norm = np.linalg.norm(self._target,  axis=-1, keepdims=True) + eps
        cos_sim = np.sum((pred / pred_norm) * (self._target / target_norm), axis=-1)
        return np.array(np.mean(1.0 - cos_sim))

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        pred_node, pred = next(iter(sources.items()))
        eps = 1e-15
        N = pred.shape[0]
        pred_norm   = np.linalg.norm(pred,         axis=-1, keepdims=True) + eps
        target_norm = np.linalg.norm(self._target, axis=-1, keepdims=True) + eps
        p_hat = pred         / pred_norm
        t_hat = self._target / target_norm
        cos_sim = np.sum(p_hat * t_hat, axis=-1, keepdims=True)
        grad = -(t_hat - p_hat * cos_sim) / (pred_norm * N)
        return {pred_node: grad}
