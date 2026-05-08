from __future__ import annotations
import numpy as np
from ..core.diffable import Diffable


class AvgPool2D(Diffable):
    """2-D average pooling over spatial dimensions.

    Source: x of shape (N, H, W, C)
    Output: (N, out_H, out_W, C)
    """

    def __init__(self, x: Diffable, pool_size: tuple = (2, 2),
                 strides: tuple = None, padding: str = 'valid'):
        self._pool_size = pool_size
        self._strides = strides if strides is not None else pool_size
        self._padding = padding.lower()
        super().__init__(x)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        x = next(iter(sources.values()))
        N, H, W, C = x.shape
        pH, pW = self._pool_size
        sH, sW = self._strides

        if self._padding == 'same':
            pad_h = max(pH - sH, 0)
            pad_w = max(pW - sW, 0)
            x = np.pad(x, ((0, 0), (pad_h // 2, pad_h - pad_h // 2),
                           (pad_w // 2, pad_w - pad_w // 2), (0, 0)))
            N, H, W, _ = x.shape

        out_H = (H - pH) // sH + 1
        out_W = (W - pW) // sW + 1

        shape = (N, out_H, out_W, pH, pW, C)
        strides = (x.strides[0], x.strides[1] * sH, x.strides[2] * sW,
                   x.strides[1], x.strides[2], x.strides[3])
        x_strided = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        return x_strided.mean(axis=(3, 4))

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        return {node: np.zeros_like(val) for node, val in sources.items()}


class GlobalAvgPool2D(Diffable):
    """Global average pooling: averages spatial dimensions per channel.

    Source: x of shape (N, H, W, C)
    Output: (N, C)
    """

    def __init__(self, x: Diffable):
        super().__init__(x)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        x = next(iter(sources.values()))
        return x.mean(axis=(1, 2))

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        return {node: np.zeros_like(val) for node, val in sources.items()}
