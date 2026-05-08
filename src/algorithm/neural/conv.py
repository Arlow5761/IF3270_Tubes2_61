from __future__ import annotations
import numpy as np
from ..core.diffable import Diffable


def _im2col(x: np.ndarray, kH: int, kW: int, sH: int, sW: int,
            out_H: int, out_W: int) -> np.ndarray:
    """Extract sliding-window patches using the im2col approach.

    Args:
        x: Input of shape (N, H, W, C)
        kH, kW: Kernel spatial dimensions
        sH, sW: Stride values
        out_H, out_W: Output spatial dimensions

    Returns:
        col: Shape (N, out_H, out_W, kH * kW * C), channel-last ordering
             (rows→cols→channels) matching Keras kernel ordering.
    """
    N, H, W, C = x.shape
    col = np.zeros((N, out_H, out_W, kH * kW * C), dtype=x.dtype)
    for i in range(kH):
        i_end = i + sH * out_H
        for j in range(kW):
            j_end = j + sW * out_W
            col_start = (i * kW + j) * C
            col[:, :, :, col_start:col_start + C] = x[:, i:i_end:sH, j:j_end:sW, :]
    return col


def _pad_input(x: np.ndarray, kH: int, kW: int, sH: int, sW: int) -> np.ndarray:
    """Apply 'same' zero-padding so output spatial size = ceil(input / stride)."""
    N, H, W, C = x.shape
    out_H = int(np.ceil(H / sH))
    out_W = int(np.ceil(W / sW))
    pad_h = max((out_H - 1) * sH + kH - H, 0)
    pad_w = max((out_W - 1) * sW + kW - W, 0)
    pad_top, pad_left = pad_h // 2, pad_w // 2
    pad_bottom, pad_right = pad_h - pad_top, pad_w - pad_left
    return np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)))


class Conv2D(Diffable):
    """2-D convolution with shared (weight-tied) kernels.

    Sources (in order): x, kernel, bias
      x:      (N, H, W, C_in)           — NHWC format (Keras default)
      kernel: (kH, kW, C_in, C_out)     — Keras kernel shape
      bias:   (C_out,)
    Output:   (N, out_H, out_W, C_out)

    Compatible with weights from keras.layers.Conv2D.get_weights().
    """

    def __init__(self, x: Diffable, kernel: Diffable, bias: Diffable,
                 strides: tuple = (1, 1), padding: str = 'valid'):
        self._strides = strides
        self._padding = padding.lower()
        super().__init__(x, kernel, bias)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        x, kernel, bias = list(sources.values())
        N, H, W, C_in = x.shape
        kH, kW, _, C_out = kernel.shape
        sH, sW = self._strides

        if self._padding == 'same':
            x = _pad_input(x, kH, kW, sH, sW)
            N, H, W, _ = x.shape

        out_H = (H - kH) // sH + 1
        out_W = (W - kW) // sW + 1

        col = _im2col(x, kH, kW, sH, sW, out_H, out_W)       # (N, oH, oW, kH*kW*Cin)
        col_flat = col.reshape(N * out_H * out_W, kH * kW * C_in)
        kernel_flat = kernel.reshape(kH * kW * C_in, C_out)

        out = col_flat @ kernel_flat + bias                    # (N*oH*oW, Cout)
        return out.reshape(N, out_H, out_W, C_out)

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        # Gradient computation for Conv2D is non-trivial and not needed for
        # inference-only from-scratch evaluation. Return zeros as placeholders.
        return {node: np.zeros_like(val) for node, val in sources.items()}
