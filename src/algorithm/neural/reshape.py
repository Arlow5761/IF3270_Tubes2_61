from __future__ import annotations
import numpy as np
from ..core.diffable import Diffable


class Flatten(Diffable):
    """Flatten all dimensions except the batch dimension (row-major, C order).

    Source: x of shape (N, d1, d2, ..., dk)
    Output: (N, d1 * d2 * ... * dk)

    Matches keras.layers.Flatten() behaviour (C-order).
    """

    def __init__(self, x: Diffable):
        super().__init__(x)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        x = next(iter(sources.values()))
        return x.reshape(x.shape[0], -1)

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        # value = upstream gradient G, shape (N, flat_dim)
        x_node, x = next(iter(sources.items()))
        return {x_node: value.reshape(x.shape)}


class Reshape(Diffable):
    """Reshape to an arbitrary target shape (batch dimension preserved).

    Source: x of shape (N, *)
    Output: (N, *target_shape)
    """

    def __init__(self, x: Diffable, target_shape: tuple):
        self._target_shape = target_shape
        super().__init__(x)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        x = next(iter(sources.values()))
        return x.reshape((x.shape[0],) + self._target_shape)

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        x_node, x = next(iter(sources.items()))
        return {x_node: value.reshape(x.shape)}
