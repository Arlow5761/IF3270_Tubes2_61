from __future__ import annotations
import numpy as np
from ..core.diffable import Diffable


class Softmax(Diffable):
    """Softmax activation along the last axis (numerically stable)."""

    def __init__(self, x: Diffable):
        super().__init__(x)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        x = next(iter(sources.values()))
        x_shifted = x - x.max(axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        # not used: for SCCE+Softmax use the SCCE criterion which takes softmax output directly
        x_node = next(iter(sources.keys()))
        return {x_node: np.zeros_like(list(sources.values())[0])}
