from __future__ import annotations
from typing import List
import numpy as np
from ..core.optimizer import Optimizer
from ..core.diffable import Diffable


class SGD(Optimizer):
    """Stochastic Gradient Descent with optional momentum.

    Args:
        lr: Learning rate.
        momentum: Momentum factor (0 = vanilla SGD).
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self._velocity: dict = {}

    def step(self, parameters: List[Diffable], gradients: List[np.ndarray]) -> None:
        super().step(parameters, gradients)
        for param, grad in zip(parameters, gradients):
            pid = id(param)
            if self.momentum > 0.0:
                if pid not in self._velocity:
                    self._velocity[pid] = np.zeros_like(param._value)
                self._velocity[pid] = self.momentum * self._velocity[pid] + grad
                param._value = param._value - self.lr * self._velocity[pid]
            else:
                param._value = param._value - self.lr * grad
