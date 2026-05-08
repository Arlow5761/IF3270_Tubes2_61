from __future__ import annotations
from typing import List
import numpy as np
from ..core.optimizer import Optimizer
from ..core.diffable import Diffable


class Adam(Optimizer):
    """Adam optimizer (Kingma & Ba, 2015).

    Args:
        lr: Learning rate.
        beta1: Exponential decay rate for 1st moment estimates.
        beta2: Exponential decay rate for 2nd moment estimates.
        epsilon: Small constant for numerical stability.
    """

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._t = 0
        self._m: dict = {}
        self._v: dict = {}

    def step(self, parameters: List[Diffable], gradients: List[np.ndarray]) -> None:
        super().step(parameters, gradients)
        self._t += 1
        for param, grad in zip(parameters, gradients):
            pid = id(param)
            if pid not in self._m:
                self._m[pid] = np.zeros_like(param._value)
                self._v[pid] = np.zeros_like(param._value)
            self._m[pid] = self.beta1 * self._m[pid] + (1.0 - self.beta1) * grad
            self._v[pid] = self.beta2 * self._v[pid] + (1.0 - self.beta2) * grad ** 2
            m_hat = self._m[pid] / (1.0 - self.beta1 ** self._t)
            v_hat = self._v[pid] / (1.0 - self.beta2 ** self._t)
            param._value = param._value - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
