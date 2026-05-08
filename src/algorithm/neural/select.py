from __future__ import annotations
import numpy as np
from ..core.diffable import Diffable


class Select(Diffable):
    """Select a slice along a given axis using a fixed integer index.

    Useful for extracting the last timestep of an RNN output, e.g.:
        Select(rnn_out, axis=1, index=-1)  →  rnn_out[:, -1, :]

    Source: x of shape (d0, d1, ..., dk, ...)
    Output: x with axis `axis` removed (selected at `index`)
    """

    def __init__(self, x: Diffable, axis: int, index: int):
        self._axis = axis
        self._index = index
        super().__init__(x)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        x = next(iter(sources.values()))
        return np.take(x, self._index, axis=self._axis)

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        x_node, x = next(iter(sources.items()))
        grad = np.zeros_like(x)
        idx = self._index % x.shape[self._axis]
        slices = [slice(None)] * x.ndim
        slices[self._axis] = idx
        grad[tuple(slices)] = value
        return {x_node: grad}
