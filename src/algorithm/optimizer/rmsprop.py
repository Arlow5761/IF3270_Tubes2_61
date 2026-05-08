from __future__ import annotations
from typing import List
import numpy as np
from ..core.optimizer import Optimizer
from ..core.diffable import Diffable


class RMSprop(Optimizer):
    """RMSprop optimizer.

    Args:
        lr:      Learning rate.
        rho:     Decay factor for the moving average of squared gradients.
        epsilon: Small constant for numerical stability.
    """

    def __init__(self, lr: float = 1e-3, rho: float = 0.9, epsilon: float = 1e-8):
        super().__init__()
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon
        self._cache: dict = {}

    def step(self, parameters: List[Diffable], gradients: List[np.ndarray]) -> None:
        super().step(parameters, gradients)
        for param, grad in zip(parameters, gradients):
            pid = id(param)
            if pid not in self._cache:
                self._cache[pid] = np.zeros_like(param._value)
            self._cache[pid] = self.rho * self._cache[pid] + (1.0 - self.rho) * grad ** 2
            param._value = param._value - self.lr * grad / (np.sqrt(self._cache[pid]) + self.epsilon)
