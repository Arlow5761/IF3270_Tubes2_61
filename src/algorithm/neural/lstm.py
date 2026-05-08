from __future__ import annotations
import numpy as np
from ..core.diffable import Diffable

# ============================================================================
# Implementation Guide: Custom LSTM Node
# ============================================================================
# Core Instructions:
# 1. Unroll in place: The graph is static. Perform the entire sequence looping
#    internally within the `_calculate_value` method.
# 2. State Dictionary: You MUST save all intermediate tensors (cell states, hidden
#    states, gate caches, inputs) needed for manual BPTT strictly into the inherited
#    `self._state` dictionary. Do NOT assign new attributes to `self` after initialization.
# 3. Padding: Ensure the input/output batches are padded correctly to the longest
#    sequence length to make the tensor math work.
# ============================================================================


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


class LSTM(Diffable):
    """Unrolled LSTM processing a full (padded) sequence.

    Keras gate order inside the concatenated kernel: [i, f, c/g, o]
      i_t = σ(x_t @ W[:, :u]       + h_{t-1} @ U[:, :u]       + b[:u])
      f_t = σ(x_t @ W[:, u:2u]     + h_{t-1} @ U[:, u:2u]     + b[u:2u])
      g_t = tanh(x_t @ W[:, 2u:3u] + h_{t-1} @ U[:, 2u:3u]   + b[2u:3u])
      o_t = σ(x_t @ W[:, 3u:]      + h_{t-1} @ U[:, 3u:]      + b[3u:])
      c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
      h_t = o_t ⊙ tanh(c_t)

    Sources (in order): x_seq, W, U, b
      x_seq: (N, T, input_dim)       — full input sequence (may be padded)
      W:     (input_dim, 4 * units)  — Keras 'kernel'
      U:     (units, 4 * units)      — Keras 'recurrent_kernel'
      b:     (4 * units,)            — Keras 'bias'
                                       If shape (2, 4*units): rows are summed.
    Output:
      (N, T, units) when return_sequences=True
      (N, units)    when return_sequences=False

    Compatible with keras.layers.LSTM weights obtained via get_weights():
        [kernel, recurrent_kernel, bias]
    """

    def __init__(self, x_seq: Diffable, W: Diffable, U: Diffable, b: Diffable,
                 return_sequences: bool = True):
        self._return_sequences = return_sequences
        super().__init__(x_seq, W, U, b)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def _calculate_value(self, sources: dict) -> np.ndarray:
        x_seq, W, U, b = list(sources.values())

        # Handle Keras implementation=2 bias shape (2, 4*units)
        if b.ndim == 2:
            b = b.sum(axis=0)

        N, T, input_dim = x_seq.shape
        units = U.shape[0]

        h = np.zeros((N, units), dtype=np.float64)
        c = np.zeros((N, units), dtype=np.float64)

        h_list, c_list = [], []
        i_list, f_list, g_list, o_list = [], [], [], []

        for t in range(T):
            x_t = x_seq[:, t, :].astype(np.float64)

            # Pre-activation: all four gates in one matmul
            z = x_t @ W + h @ U + b                  # (N, 4*units)

            i_t = _sigmoid(z[:, :units])              # input gate
            f_t = _sigmoid(z[:, units:2 * units])     # forget gate
            g_t = np.tanh(z[:, 2 * units:3 * units]) # cell gate
            o_t = _sigmoid(z[:, 3 * units:])          # output gate

            c = f_t * c + i_t * g_t                  # cell state
            h = o_t * np.tanh(c)                     # hidden state

            h_list.append(h.copy()); c_list.append(c.copy())
            i_list.append(i_t); f_list.append(f_t)
            g_list.append(g_t); o_list.append(o_t)

        # Cache everything needed for BPTT
        self._state['h_seq'] = np.stack(h_list, axis=0)   # (T, N, units)
        self._state['c_seq'] = np.stack(c_list, axis=0)
        self._state['i_seq'] = np.stack(i_list, axis=0)
        self._state['f_seq'] = np.stack(f_list, axis=0)
        self._state['g_seq'] = np.stack(g_list, axis=0)
        self._state['o_seq'] = np.stack(o_list, axis=0)
        self._state['b_used'] = b                         # (possibly summed) bias

        if self._return_sequences:
            return np.stack(h_list, axis=1)              # (N, T, units)
        else:
            return h_list[-1]                            # (N, units)

    # ── Backward pass (BPTT) ──────────────────────────────────────────────────

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        """Backpropagation Through Time for LSTM.

        Args:
            sources: dict mapping nodes → their forward values.
            value:   upstream gradient G = dL/d_output.
                     Shape (N, T, units) if return_sequences, else (N, units).
        """
        x_seq, W, U, b_raw = list(sources.values())
        x_node, W_node, U_node, b_node = list(sources.keys())

        h_seq = self._state['h_seq']   # (T, N, units)
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
            h_t = h_seq[t];  c_t = c_seq[t]
            c_prev = c_seq[t - 1] if t > 0 else np.zeros((N, units))
            h_prev = h_seq[t - 1] if t > 0 else np.zeros((N, units))
            x_t    = x_seq[:, t, :].astype(np.float64)
            i_t = i_seq[t]; f_t = f_seq[t]; g_t = g_seq[t]; o_t = o_seq[t]

            # ── Gradient into h_t ────────────────────────────────────────
            dh = G[:, t, :] + dh_next                          # (N, units)

            # h_t = o_t * tanh(c_t)
            tanh_ct = np.tanh(c_t)
            do      = dh * tanh_ct                             # (N, units)
            dc      = dh * o_t * (1.0 - tanh_ct ** 2) + dc_next

            # c_t = f_t * c_prev + i_t * g_t
            df      = dc * c_prev
            di      = dc * g_t
            dg      = dc * i_t
            dc_next = dc * f_t

            # Gate pre-activation gradients (chain through sigmoid / tanh)
            di_pre = di * i_t * (1.0 - i_t)                   # (N, units)
            df_pre = df * f_t * (1.0 - f_t)
            dg_pre = dg * (1.0 - g_t ** 2)
            do_pre = do * o_t * (1.0 - o_t)

            dz = np.concatenate([di_pre, df_pre, dg_pre, do_pre], axis=-1)  # (N, 4u)

            dW      += x_t.T @ dz                              # (input_dim, 4u)
            dU      += h_prev.T @ dz                           # (units, 4u)
            db      += dz.sum(axis=0)                          # (4u,)
            dx_seq[:, t, :] = dz @ W.T                        # (N, input_dim)
            dh_next = dz @ U.T                                 # (N, units)

        # If original bias was (2, 4u), gradient must match that shape
        if b_raw.ndim == 2:
            db = np.stack([db, db], axis=0)                    # (2, 4u) — equal split

        return {x_node: dx_seq, W_node: dW, U_node: dU, b_node: db}
