from __future__ import annotations
from abc import ABC, ABCMeta, abstractmethod
import numpy as np
import warnings


class _DiffableMeta(ABCMeta):
    """
    Metaclass to detect when a Diffable object and all its subclasses 
    have completely finished their __init__ phase.
    """

    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj._init_complete = True
        return obj


class Diffable(ABC, metaclass=_DiffableMeta):
    """
    Abstract base class for all objects that can be auto-differentiated.

    Diffable represents nodes in a computational graph used for automatic differentiation.
    Each node has sources (inputs) and sinks (outputs), forming a directed acyclic graph (DAG).
    The graph supports forward pass (value computation) and backward pass (gradient computation).

    Attributes:
        _sources (dict[Diffable, np.ndarray | None]): Dictionary mapping source nodes to their cached gradients.
        _sinks (list[Diffable]): List of sink nodes that depend on this node.
        _value (np.ndarray | None): Cached computed value of this node.
        _warn_on_no_sources (bool): Class attribute to control warnings for nodes with no sources.
        _warn_on_unmanaged_state (bool): Class attribute to control warnings for nodes with unmanaged state.
    """


    # Leaf nodes should set this property to false
    # to disable warnings on object initialization
    _warn_on_no_sources = True

    # Nodes with complex state management should set
    # this property to false to disable warnings when
    # assigning state as an object property
    _warn_on_unmanaged_state = True


    def __init__(self, *sources: Diffable):
        """
        Initialize a Diffable node with its source dependencies.

        Args:
            *sources: Variable number of Diffable objects that this node depends on.
                      These form the inputs to this node in the computational graph.

        The node registers itself as a sink in each source node. If no sources are provided
        and _warn_on_no_sources is True, a warning is issued for potential disconnected nodes.
        """
        self._sources: dict[Diffable, np.ndarray | None] = {source: None for source in sources}
        self._sinks: list[Diffable] = []
        self._value: np.ndarray | None = None
        self._state: dict = {}

        if len(sources) == 0 and self._warn_on_no_sources:
            warnings.warn(
                f"Node '{self.__class__.__name__}' was initialized with no sources. "
                f"If this is an operation node, it will be disconnected from the graph.",
                category=UserWarning,
                stacklevel=2,
            )

        for source in sources:
            source._add_sink(self)
    

    def __setattr__(self, name, value):
        """
        Intercepts attribute assignment to warn about potential hidden state bugs.
        """

        if self._warn_on_unmanaged_state and getattr(self, '_init_complete', False):
            if name not in ('_value', '_sources', '_sinks', '_state'):
                warnings.warn(
                    f"Potential state leak in '{self.__class__.__name__}': "
                    f"Assigned to 'self.{name}' after initialization.\n"
                    f"-> To ensure automatic clearing between iterations, "
                    f"store intermediate caches in the state dictionary: `self._state['{name}'] = ...`",
                    category=UserWarning,
                    stacklevel=2,
                )
        
        super().__setattr__(name, value)
    

    def detach(self):
        """
        Detaches the current node and subsequent ones from the computation graph
        """

        for sink in self._sinks:
            sink.detach()

        self._sinks.clear()
        
        for source in self._sources:
            source._rem_sink(self)


    def _add_source(self, source: Diffable):
        """
        Add a source dependency to this node.

        Args:
            source: The Diffable node to add as a source.
        """
        self._sources[source] = None


    def _rem_source(self, source: Diffable):
        """
        Remove a source dependency from this node.

        Args:
            source: The Diffable node to remove as a source.
        """
        self._sources.pop(source)


    def _add_sink(self, sink: Diffable):
        """
        Add a sink dependency to this node.

        Args:
            sink: The Diffable node that depends on this node.
        """
        self._sinks.append(sink)


    def _rem_sink(self, sink: Diffable):
        """
        Remove a sink dependency from this node.

        Args:
            sink: The Diffable node to remove as a sink.
        """
        self._sinks.remove(sink)
    

    def clear_values(self):
        """
        Clear the cached value and recursively clear values in the graph.

        This method resets the _value of this node and all its sinks,
        forcing recomputation on the next get_value() call.
        """
        self._value = None

        self._state.clear()

        for source in self._sources:
            self._sources[source] = None

        for sink in self._sinks:
            sink.clear_values()


    def clear_gradients_backwards(self):
        """
        Clear cached gradients backwards through the sources.

        This recursively clears gradients from this node back to the sources,
        preparing for a new backward pass.
        """
        for source in self._sources:
            self._sources[source] = None
            source.clear_gradients_backwards()
    

    def _clear_gradients_forwards(self):
        """
        Clear cached gradients forwards through the sinks.

        This recursively clears gradients from this node forward to the sinks.
        """
        for source in self._sources:
            self._sources[source] = None
        
        for sink in self._sinks:
            sink._clear_gradients_forwards()
    

    def clear_gradients(self):
        """
        Clear all cached gradients in the computation graph.

        This combines backward and forward clearing to reset the entire graph's gradients.
        """
        self.clear_gradients_backwards()
    
        for sink in self._sinks:
            sink._clear_gradients_forwards()
    

    def get_value(self) -> np.ndarray:
        """
        Compute and return the value of this node.

        If the value is not cached, it recursively computes values for sources,
        then calls _calculate_value to compute this node's value.

        Returns:
            The computed numpy array value of this node.
        """
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
        """
        Compute the gradient of this node with respect to the target node.

        This performs backpropagation through the computational graph.
        If target is this node, returns ones. Otherwise, recursively computes
        gradients through the sinks.

        Args:
            target: The Diffable node to compute gradient with respect to.

        Returns:
            The gradient as a numpy array.

        Raises:
            ValueError: If the target is not found in the computation graph.
        """
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
        """
        Abstract method to compute the value of this node.

        Subclasses must implement this to define how the node's value is calculated
        from its source values.

        Args:
            sources: Dictionary mapping source Diffable nodes to their numpy array values.

        Returns:
            The computed value as a numpy array.
        """
        raise NotImplementedError("calculate value method not implemented")
    
    
    @abstractmethod
    def _calculate_gradient(self, sources: dict[Diffable, np.ndarray], value: np.ndarray) -> dict[Diffable, np.ndarray]:
        """
        Abstract method to compute gradients with respect to sources.

        Subclasses must implement this to define how gradients flow backward
        through this node.

        Args:
            sources: Dictionary mapping source Diffable nodes to their numpy array values.
            value: The gradient flowing into this node from the sink.

        Returns:
            Dictionary mapping source Diffable nodes to their computed gradients.
        """
        raise NotImplementedError("calculate gradient method not implemented")