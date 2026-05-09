from __future__ import annotations
import numpy as np
from ..core.diffable import Diffable


class Embedding(Diffable):
    """Token embedding lookup table.

    Sources (in order): indices, matrix
      indices: integer array of shape (N, T) or (N,)
      matrix:  embedding matrix of shape (vocab_size, embed_dim)
    Output: matrix[indices], shape (..., embed_dim)

    Compatible with weights from keras.layers.Embedding.get_weights()[0].
    """

    def __init__(self, indices: Diffable, matrix: Diffable):
        super().__init__(indices, matrix)

    def _calculate_value(self, sources: dict) -> np.ndarray:
        indices, matrix = list(sources.values())
        return matrix[indices.astype(int)]

    def _calculate_gradient(self, sources: dict, value: np.ndarray) -> dict:
        indices_node, matrix_node = list(sources.keys())
        indices, matrix = list(sources.values())
        d_matrix = np.zeros_like(matrix)
        flat_idx = indices.astype(int).ravel()
        flat_G   = value.reshape(-1, matrix.shape[-1])
        np.add.at(d_matrix, flat_idx, flat_G)
        return {indices_node: np.zeros_like(indices, dtype=float),
                matrix_node:  d_matrix}
