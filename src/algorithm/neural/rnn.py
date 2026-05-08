from __future__ import annotations
import numpy as np
from ..core.diffable import Diffable

# ============================================================================
# Implementation Guide: Custom RNN Node
# ============================================================================
# Core Instructions:
# 1. Unroll in place: The graph is static. Perform the entire sequence looping
#    internally within the `_calculate_value` method.
# 2. State Dictionary: You MUST save all intermediate tensors (hidden states, inputs)
#    needed for manual BPTT strictly into the inherited `self._state` dictionary.
#    Do NOT assign new attributes to `self` after initialization.
# 3. Padding: Ensure the input/output batches are padded correctly to the longest
#    sequence length to make the tensor math work.
# ============================================================================


class SimpleRNN(Diffable):
    """Unrolled SimpleRNN processing a full (padded) sequence.

    h_t = tanh(x_t @ W_x + h_{t-1} @ W_h + b)

    Sources (in order): x_seq, W_x, W_h, b
      x_seq: (N, T, input_dim)   — full input sequence (may be padded)
      W_x:   (input_dim, units)  — Keras 'kernel'
      W_h:   (units, units)      — Keras 'recurrent_kernel'
      b:     (units,)            — Keras 'bias'
    Output:
      (N, T, units) when return_sequences=True
      (N, units)    when return_sequences=False

    Compatible with keras.layers.SimpleRNN weights obtained via get_weights():
        [kernel, recurrent_kernel, bias]
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
            x_t = x_seq[:, t, :]                            # (N, input_dim)
            h   = np.tanh(x_t @ W_x + h @ W_h + b)         # (N, units)
            h_all.append(h.copy())

        # Cache all hidden states for BPTT (indexed as [t, n, units])
        self._state['h_seq'] = np.stack(h_all, axis=0)      # (T, N, units)
        self._state['x_seq'] = x_seq                        # keep reference

        if self._return_sequences:
            return np.stack(h_all, axis=1)                  # (N, T, units)
        else:
            return h_all[-1]                                # (N, units)

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        """BPTT through time for SimpleRNN.

        Args:
            sources: dict mapping nodes to their forward values.
            value:   upstream gradient G (dL/d_output).
                     Shape (N, T, units) if return_sequences, else (N, units).
        """
        x_seq, W_x, W_h, b = list(sources.values())
        x_node, W_x_node, W_h_node, b_node = list(sources.keys())

        h_seq = self._state['h_seq']          # (T, N, units)
        N, T, input_dim = x_seq.shape
        units = W_h.shape[0]

        G = value
        if not self._return_sequences:
            # Expand scalar gradient to sequence: only last step has non-zero grad
            G_full = np.zeros((N, T, units), dtype=x_seq.dtype)
            G_full[:, -1, :] = G
            G = G_full

        dW_x   = np.zeros_like(W_x)
        dW_h   = np.zeros_like(W_h)
        db     = np.zeros_like(b)
        dx_seq = np.zeros_like(x_seq)
        dh_next = np.zeros((N, units), dtype=x_seq.dtype)

        for t in reversed(range(T)):
            h_t    = h_seq[t]                                     # (N, units)
            h_prev = h_seq[t - 1] if t > 0 else np.zeros_like(h_t)
            x_t    = x_seq[:, t, :]                               # (N, input_dim)

            # Total gradient flowing into h_t
            dh = G[:, t, :] + dh_next                             # (N, units)

            # Backward through tanh: h_t = tanh(z_t), dz_t = dh * (1 - h_t^2)
            dtanh = dh * (1.0 - h_t ** 2)                        # (N, units)

            dW_x   += x_t.T @ dtanh                              # (input_dim, units)
            dW_h   += h_prev.T @ dtanh                           # (units, units)
            db     += dtanh.sum(axis=0)                          # (units,)
            dx_seq[:, t, :] = dtanh @ W_x.T                     # (N, input_dim)
            dh_next = dtanh @ W_h.T                              # (N, units)

        return {x_node:   dx_seq,
                W_x_node: dW_x,
                W_h_node: dW_h,
                b_node:   db}
