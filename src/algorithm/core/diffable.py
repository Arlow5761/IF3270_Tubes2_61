from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class Diffable(ABC):
    """
    Abstract base class for all objects that can be auto-differentiated
    """


    def __init__(self):
        self._sources: dict[Diffable, np.ndarray | None] = {}
        self._sinks: list[Diffable] = []
        self._value: np.ndarray | None = None


    def _add_source(self, source: Diffable):
        self._sources[source] = None


    def _rem_source(self, source: Diffable):
        self._sources.pop(source)


    def _add_sink(self, sink: Diffable):
        self._sinks.append(sink)


    def _rem_sink(self, sink: Diffable):
        self._sinks.remove(sink)
    

    def clear_values(self):
        self._value = None

        for source in self._sources:
            self._sources[source] = None

        for sink in self._sinks:
            sink.clear_values()


    def clear_gradients_backwards(self):
        for source in self._sources:
            self._sources[source] = None
            source.clear_gradients_backwards()
    

    def _clear_gradients_forwards(self):
        for source in self._sources:
            self._sources[source] = None
        
        for sink in self._sinks:
            sink._clear_gradients_forwards()
    

    def clear_gradients(self):
        self.clear_gradients_backwards()
    
        for sink in self._sinks:
            sink._clear_gradients_forwards()
    

    def get_value(self) -> np.ndarray:
        value_stale = self._value is None

        if not value_stale:
            for source in self._sources:
                if source._value is None:
                    value_stale = True
                    source.get_value()
        
        if value_stale:
            self.clear_values()
            self.clear_gradients_backwards()

            source_values = {source: source._value for source in self._sources}
            self._value = self._calculate_value(source_values)

        return self._value
    

    def get_gradient(self, target: Diffable) -> np.ndarray:
        self.get_value() # Ensure value is present
        
        if target == self:
            return np.ones_like(self._value)
        
        if not self._sinks:
            raise ValueError("differentiator not found in computation graph")

        total_gradient = np.zeros_like(self._value)
        gradient_valid = False

        for sink in self._sinks:
            try:
                gradients_stale = False

                for gradient in sink._sources.values():
                    if gradient is None:
                        gradients_stale = True

                if gradients_stale:
                    sink_value = sink.get_value()
                    source_values = {source: source._value for source in sink._sources}
                    sink._sources = sink._calculate_gradient(source_values, sink_value)

                total_gradient += sink._sources[self] * sink.get_gradient(target)
                gradient_valid = True
            except ValueError:
                pass
        
        if gradient_valid:
            return total_gradient
        else:
            raise ValueError("differentiator not found in computation graph")


    @abstractmethod
    def _calculate_value(self, sources: dict[Diffable, np.ndarray]) -> np.ndarray:
        raise NotImplementedError("calculate value method not implemented")
    
    
    @abstractmethod
    def _calculate_gradient(self, sources: dict[Diffable, np.ndarray], value: np.ndarray) -> dict[Diffable, np.ndarray]:
        raise NotImplementedError("calculate gradient method not implemented")