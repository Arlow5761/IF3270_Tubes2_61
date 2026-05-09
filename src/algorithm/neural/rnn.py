from __future__ import annotations
import numpy as np
from ..core.diffable import Diffable


class SimpleRNN(Diffable):
    """Unrolled SimpleRNN: h_t = tanh(x_t @ W_x + h_{t-1} @ W_h + b).

    Sources: x_seq (N,T,in), W_x (in,u), W_h (u,u), b (u,)
    Output: (N,T,u) if return_sequences else (N,u)
    Compatible with keras.layers.SimpleRNN.get_weights(): [kernel, recurrent_kernel, bias]
    """

    def __init__(self, x_seq: Diffable, W_x: Diffable, W_h: Diffable, b: Diffable,
                 return_sequences: bool = True):
        self._return_sequences = return_sequences
        super().__init__(x_seq, W_x, W_h, b)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        x_seq, W_x, W_h, b = list(sources.values())
        N, T, input_dim = x_seq.shape
        units = W_h.shape[0]

        h = np.zeros((N, units), dtype=x_seq.dtype)
        h_all = []

        for t in range(T):
            x_t = x_seq[:, t, :]
            h   = np.tanh(x_t @ W_x + h @ W_h + b)
            h_all.append(h.copy())

        self._state['h_seq'] = np.stack(h_all, axis=0)
        self._state['x_seq'] = x_seq

        if self._return_sequences:
            return np.stack(h_all, axis=1)
        else:
            return h_all[-1]

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        x_seq, W_x, W_h, b = list(sources.values())
        x_node, W_x_node, W_h_node, b_node = list(sources.keys())

        h_seq = self._state['h_seq']
        N, T, input_dim = x_seq.shape
        units = W_h.shape[0]

        G = value
        if not self._return_sequences:
            G_full = np.zeros((N, T, units), dtype=x_seq.dtype)
            G_full[:, -1, :] = G
            G = G_full

        dW_x    = np.zeros_like(W_x)
        dW_h    = np.zeros_like(W_h)
        db      = np.zeros_like(b)
        dx_seq  = np.zeros_like(x_seq)
        dh_next = np.zeros((N, units), dtype=x_seq.dtype)

        for t in reversed(range(T)):
            h_t    = h_seq[t]
            h_prev = h_seq[t - 1] if t > 0 else np.zeros_like(h_t)
            x_t    = x_seq[:, t, :]
            dh     = G[:, t, :] + dh_next
            dtanh  = dh * (1.0 - h_t ** 2)

            dW_x        += x_t.T @ dtanh
            dW_h        += h_prev.T @ dtanh
            db          += dtanh.sum(axis=0)
            dx_seq[:, t, :] = dtanh @ W_x.T
            dh_next      = dtanh @ W_h.T

        return {x_node:   dx_seq,
                W_x_node: dW_x,
                W_h_node: dW_h,
                b_node:   db}
