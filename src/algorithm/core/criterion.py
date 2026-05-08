from .diffable import Diffable
import numpy as np

class Criterion(Diffable):
    """
    Base class for criterion nodes
    """

    def __init__(self, output: Diffable, target: np.ndarray):
        super().__init__(output)
        self._target = target