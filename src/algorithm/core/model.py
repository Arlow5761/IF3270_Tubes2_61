from __future__ import annotations
from typing import List, Dict, Any, Type
import numpy as np
from .diffable import Diffable
from .criterion import Criterion
from .optimizer import Optimizer


class Model:
    """
    A model class similar to PyTorch's nn.Module, where subclasses build the computational graph
    in __init__ by registering inputs and outputs. The base class infers parameters by traversing
    the graph, handles training logic, loss history, and storage of latest layer outputs for visualization.
    """

    def __init__(self):
        self.inputs: List[Diffable] = []
        self.output: Diffable | None = None
        self.parameters: List[Diffable] = []
        self.loss_history: List[float] = []
        self.latest_outputs: Dict[str, np.ndarray] = {}

    def register_input(self, input_diffable: Diffable) -> None:
        """
        Register an input Diffable to the model.

        Args:
            input_diffable: The Diffable representing an input to the model.
        """
        self.inputs.append(input_diffable)

    def register_output(self, output_diffable: Diffable) -> None:
        """
        Register the output Diffable of the model and infer parameters.

        Args:
            output_diffable: The Diffable representing the output of the model.
        """
        self.output = output_diffable
        self.parameters = self._infer_parameters()

    def _infer_parameters(self) -> List[Diffable]:
        """
        Infer trainable parameters by traversing the graph from output to inputs,
        collecting all leaf nodes (no sources) that are not registered inputs.

        Returns:
            List of inferred parameter Diffables.
        """
        if self.output is None:
            return []

        to_visit = [self.output]
        params = []

        while to_visit:
            node = to_visit.pop()

            if node not in self.inputs and not node._sources:
                params.append(node)

            for source in node._sources:
                to_visit.append(source)

        return params

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass by setting the input value and computing the output.

        Args:
            x: Input data as numpy array.

        Returns:
            Output of the model as numpy array.

        Raises:
            ValueError: If the model does not have exactly one input or no output registered.
        """
        if len(self.inputs) != 1:
            raise ValueError("Model must have exactly one input registered")
        if self.output is None:
            raise ValueError("Model must have an output registered")

        # Assume the input Diffable allows setting its value (e.g., Input class)
        self.inputs[0]._value = x
        return self.output.get_value()

    def get_parameters(self) -> List[Diffable]:
        """
        Get all inferred trainable parameters.

        Returns:
            List of Diffable parameters.
        """
        return self.parameters

    def train_step(self, x: np.ndarray, y: np.ndarray, criterion_class: Type[Criterion], optimizer: Optimizer) -> float:
        """
        Perform a single training step: forward pass, loss computation, backpropagation, and parameter update.

        Args:
            x: Input data as numpy array.
            y: Target data as numpy array.
            criterion_class: A Criterion class that will be instantiated with (output, target).
            optimizer: An Optimizer instance to update parameters based on gradients.

        Returns:
            Loss value for this step.
        """
        output = self.forward(x)
        
        # Create criterion instance and attach it to the graph
        loss = criterion_class(self.output, y)
        loss_value = loss.get_value()
        self.loss_history.append(loss_value)

        # Compute gradients and update parameters
        grads = [loss.get_gradient(param) for param in self.parameters]
        optimizer.step(self.parameters, grads)
        
        # Detach criterion from graph
        loss.detach()
        
        return loss_value

    def get_loss_history(self) -> List[float]:
        """
        Get the history of loss values from training steps.

        Returns:
            List of loss values.
        """
        return self.loss_history

    def get_latest_outputs(self) -> Dict[str, np.ndarray]:
        """
        Get the latest outputs from each layer after the most recent forward pass.
        Subclasses should populate this in their forward method or elsewhere for visualization.

        Returns:
            Dictionary mapping layer names to their outputs.
        """
        return self.latest_outputs

    def clear_history(self) -> None:
        """
        Clear the loss history.
        """
        self.loss_history = []
        return self.latest_outputs