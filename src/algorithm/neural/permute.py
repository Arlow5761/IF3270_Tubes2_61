from __future__ import annotations
import numpy as np
from ..core.diffable import Diffable


class Permute(Diffable):
    """Permute (transpose) the axes of a tensor.

    Source: x of arbitrary shape
    Output: x transposed according to `axes`

    Example — NHWC → NCHW: Permute(x, (0, 3, 1, 2))
    """

    def __init__(self, x: Diffable, axes: tuple):
        self._axes = tuple(axes)
        self._inv_axes = tuple(np.argsort(axes))
        super().__init__(x)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        x = next(iter(sources.values()))
        return np.transpose(x, self._axes)

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        x_node = next(iter(sources.keys()))
        return {x_node: np.transpose(value, self._inv_axes)}
