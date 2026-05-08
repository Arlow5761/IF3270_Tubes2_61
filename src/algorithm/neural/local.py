from __future__ import annotations
import numpy as np
from ..core.diffable import Diffable
from .conv import _im2col, _pad_input


class LocallyConnected2D(Diffable):
    """2-D locally connected (non-shared) layer.

    Each output spatial position has its own independent kernel, so parameters
    are NOT shared across positions (unlike Conv2D).

    Sources (in order): x, kernel, bias
      x:      (N, H, W, C_in)
      kernel: (out_H * out_W, kH * kW * C_in, C_out)  — Keras LC2D kernel shape
      bias:   (out_H * out_W, C_out)
    Output:   (N, out_H, out_W, C_out)

    Compatible with weights from keras.layers.LocallyConnected2D.get_weights().
    """

    def __init__(self, x: Diffable, kernel: Diffable, bias: Diffable,
                 kernel_size: tuple = (3, 3), strides: tuple = (1, 1),
                 padding: str = 'valid'):
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding.lower()
        super().__init__(x, kernel, bias)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        x, kernel, bias = list(sources.values())
        N, H, W, C_in = x.shape
        kH, kW = self._kernel_size
        sH, sW = self._strides
        n_pos, klen, C_out = kernel.shape  # klen = kH * kW * C_in

        if self._padding == 'same':
            x = _pad_input(x, kH, kW, sH, sW)
            N, H, W, _ = x.shape

        out_H = (H - kH) // sH + 1
        out_W = (W - kW) // sW + 1

        col = _im2col(x, kH, kW, sH, sW, out_H, out_W)   # (N, oH, oW, kH*kW*Cin)
        col = col.reshape(N, out_H * out_W, klen)          # (N, n_pos, klen)

        # Per-position matmul: (N, n_pos, klen) @ (n_pos, klen, Cout) → (N, n_pos, Cout)
        out = np.einsum('npi,pio->npo', col, kernel) + bias[np.newaxis, :, :]
        return out.reshape(N, out_H, out_W, C_out)

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        return {node: np.zeros_like(val) for node, val in sources.items()}
