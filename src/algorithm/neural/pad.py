from __future__ import annotations
import numpy as np
from ..core.diffable import Diffable


class ZeroPad2D(Diffable):
    """Add zero-padding to the spatial dimensions of a 4-D tensor.

    Source: x of shape (N, H, W, C)
    Output: (N, H + pad_top + pad_bottom, W + pad_left + pad_right, C)

    padding can be:
      - int: same padding on all 4 sides
      - (int, int): (height_pad, width_pad) — symmetric per axis
      - ((top, bottom), (left, right)): explicit per-side padding
    """

    def __init__(self, x: Diffable, padding=1):
        if isinstance(padding, int):
            self._padding = ((padding, padding), (padding, padding))
        elif len(padding) == 2 and isinstance(padding[0], int):
            self._padding = ((padding[0], padding[0]), (padding[1], padding[1]))
        else:
            self._padding = tuple(tuple(p) for p in padding)
        super().__init__(x)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        x = next(iter(sources.values()))
        (pt, pb), (pl, pr) = self._padding
        return np.pad(x, ((0, 0), (pt, pb), (pl, pr), (0, 0)))

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        x_node, x = next(iter(sources.items()))
        (pt, pb), (pl, pr) = self._padding
        N, H, W, C = x.shape
        return {x_node: value[:, pt:pt + H, pl:pl + W, :]}
