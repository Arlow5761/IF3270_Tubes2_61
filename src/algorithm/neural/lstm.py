from __future__ import annotations
import numpy as np
from ..core.diffable import Diffable


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


class LSTM(Diffable):
    """Unrolled LSTM with Keras gate order [i, f, g, o].

    Sources: x_seq (N,T,in), W (in,4u), U (u,4u), b (4u,) or (2,4u)
    Output: (N,T,u) if return_sequences else (N,u)
    Compatible with keras.layers.LSTM.get_weights(): [kernel, recurrent_kernel, bias]
    """

    def __init__(self, x_seq: Diffable, W: Diffable, U: Diffable, b: Diffable,
                 return_sequences: bool = True):
        self._return_sequences = return_sequences
        super().__init__(x_seq, W, U, b)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        x_seq, W, U, b = list(sources.values())

        if b.ndim == 2:  # Keras implementation=2 bias shape (2, 4*units)
            b = b.sum(axis=0)

        N, T, input_dim = x_seq.shape
        units = U.shape[0]

        h = np.zeros((N, units), dtype=np.float64)
        c = np.zeros((N, units), dtype=np.float64)

        h_list, c_list = [], []
        i_list, f_list, g_list, o_list = [], [], [], []

        for t in range(T):
            x_t = x_seq[:, t, :].astype(np.float64)
            z   = x_t @ W + h @ U + b

            i_t = _sigmoid(z[:, :units])
            f_t = _sigmoid(z[:, units:2 * units])
            g_t = np.tanh(z[:, 2 * units:3 * units])
            o_t = _sigmoid(z[:, 3 * units:])

            c = f_t * c + i_t * g_t
            h = o_t * np.tanh(c)

            h_list.append(h.copy()); c_list.append(c.copy())
            i_list.append(i_t); f_list.append(f_t)
            g_list.append(g_t); o_list.append(o_t)

        self._state['h_seq'] = np.stack(h_list, axis=0)
        self._state['c_seq'] = np.stack(c_list, axis=0)
        self._state['i_seq'] = np.stack(i_list, axis=0)
        self._state['f_seq'] = np.stack(f_list, axis=0)
        self._state['g_seq'] = np.stack(g_list, axis=0)
        self._state['o_seq'] = np.stack(o_list, axis=0)
        self._state['b_used'] = b

        if self._return_sequences:
            return np.stack(h_list, axis=1)
        else:
            return h_list[-1]

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        x_seq, W, U, b_raw = list(sources.values())
        x_node, W_node, U_node, b_node = list(sources.keys())

        h_seq = self._state['h_seq']
        c_seq = self._state['c_seq']
        i_seq = self._state['i_seq']
        f_seq = self._state['f_seq']
        g_seq = self._state['g_seq']
        o_seq = self._state['o_seq']

        N, T, input_dim = x_seq.shape
        units = U.shape[0]

        G = value
        if not self._return_sequences:
            G_full = np.zeros((N, T, units), dtype=np.float64)
            G_full[:, -1, :] = G
            G = G_full

        dW     = np.zeros_like(W, dtype=np.float64)
        dU     = np.zeros_like(U, dtype=np.float64)
        db     = np.zeros(4 * units, dtype=np.float64)
        dx_seq = np.zeros_like(x_seq, dtype=np.float64)

        dh_next = np.zeros((N, units), dtype=np.float64)
        dc_next = np.zeros((N, units), dtype=np.float64)

        for t in reversed(range(T)):
            h_t    = h_seq[t]; c_t = c_seq[t]
            c_prev = c_seq[t - 1] if t > 0 else np.zeros((N, units))
            h_prev = h_seq[t - 1] if t > 0 else np.zeros((N, units))
            x_t    = x_seq[:, t, :].astype(np.float64)
            i_t = i_seq[t]; f_t = f_seq[t]; g_t = g_seq[t]; o_t = o_seq[t]

            dh      = G[:, t, :] + dh_next
            tanh_ct = np.tanh(c_t)
            do      = dh * tanh_ct
            dc      = dh * o_t * (1.0 - tanh_ct ** 2) + dc_next

            df      = dc * c_prev
            di      = dc * g_t
            dg      = dc * i_t
            dc_next = dc * f_t

            di_pre = di * i_t * (1.0 - i_t)
            df_pre = df * f_t * (1.0 - f_t)
            dg_pre = dg * (1.0 - g_t ** 2)
            do_pre = do * o_t * (1.0 - o_t)

            dz = np.concatenate([di_pre, df_pre, dg_pre, do_pre], axis=-1)

            dW             += x_t.T @ dz
            dU             += h_prev.T @ dz
            db             += dz.sum(axis=0)
            dx_seq[:, t, :] = dz @ W.T
            dh_next         = dz @ U.T

        if b_raw.ndim == 2:  # restore (2, 4u) shape to match original bias
            db = np.stack([db, db], axis=0)

        return {x_node: dx_seq, W_node: dW, U_node: dU, b_node: db}
