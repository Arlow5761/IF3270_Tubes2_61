from __future__ import annotations
import numpy as np
from ..core.diffable import Diffable


class Concatenate(Diffable):
    """Concatenate multiple tensors along a specified axis.

    Sources: any number of Diffable nodes with compatible shapes.
    Output: concatenated tensor along `axis`.
    """

    def __init__(self, *sources: Diffable, axis: int = -1):
        self._axis = axis
        super().__init__(*sources)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        arrays = list(sources.values())
        return np.concatenate(arrays, axis=self._axis)

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        sizes = [v.shape[self._axis] for v in sources.values()]
        splits = np.split(value, np.cumsum(sizes[:-1]), axis=self._axis)
        return {node: grad for node, grad in zip(sources.keys(), splits)}
