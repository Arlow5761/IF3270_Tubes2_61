from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from .diffable import Diffable


class Optimizer(ABC):
    """
    Base class for optimization algorithms.

    Subclasses implement different optimization strategies (SGD, Adam, RMSprop, etc.)
    for updating model parameters based on computed gradients.
    """


    def __init__(self):
        """
        Initialize the optimizer.
        """
        pass


    @abstractmethod
    def step(self, parameters: List[Diffable], gradients: List[np.ndarray]) -> None:
        """
        Perform a single optimization step to update parameters based on gradients.

        Args:
            parameters: List of Diffable parameters to update.
            gradients: List of gradient arrays corresponding to each parameter.
                      gradients[i] is the gradient for parameters[i].

        Raises:
            ValueError: If the number of parameters and gradients don't match.
        """
        if len(parameters) != len(gradients):
            raise ValueError(
                f"Number of parameters ({len(parameters)}) must match "
                f"number of gradients ({len(gradients)})"
            )
