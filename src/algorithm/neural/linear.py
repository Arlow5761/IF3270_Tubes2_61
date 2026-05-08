from __future__ import annotations
import numpy as np
from ..core.diffable import Diffable


class Linear(Diffable):
    """Fully connected (Dense) layer: y = x @ W + b.

    Sources (in order): x, W, b
      x: (..., in_features)
      W: (in_features, out_features)   — matches Keras kernel shape
      b: (out_features,)
    Output: (..., out_features)
    """

    def __init__(self, x: Diffable, W: Diffable, b: Diffable):
        super().__init__(x, W, b)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        x, W, b = list(sources.values())
        return x @ W + b

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        # value = upstream gradient G = dL/dy, shape (..., out_features)
        # dL/dx = G @ W.T
        # dL/dW = x.T @ G  (for 2-D input; uses np.tensordot for batched)
        # dL/db = G.sum over all non-last axes
        x_node, W_node, b_node = list(sources.keys())
        x, W, _ = list(sources.values())
        G = value
        dL_dx = G @ W.T
        # Handle batched input: collapse leading dims except last
        x_2d = x.reshape(-1, x.shape[-1])
        G_2d = G.reshape(-1, G.shape[-1])
        dL_dW = x_2d.T @ G_2d
        dL_db = G_2d.sum(axis=0)
        return {x_node: dL_dx, W_node: dL_dW, b_node: dL_db}
