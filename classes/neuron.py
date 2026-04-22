from typing import Callable

import numpy as np

from classes.algorithm import Algorithm
from classes.data import Data


class Neuron:
    """A single neuron."""

    def __init__(
        self,
        name: str,
        weights: np.ndarray,
        learning_rate: float,
        bias: float = 1.0,
        activation_function: Callable[[float], float] = Algorithm.sigmoid,
    ):
        self.feature = name
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate
        self.activation_function = activation_function

    def forward_propagation(self, x_input_vector: np.ndarray) -> tuple[float, float]:
        """'Observes' data and produces an activation value."""
        # Calculate pre activation value (z)
        z_pre_activation_value = np.dot(self.weights, x_input_vector) + self.bias

        # Use an activation function to produce the activation value (a)
        a_activation_value = self.activation_function(z_pre_activation_value)
        return z_pre_activation_value, a_activation_value

    def backward_propagation(
        self, a_activation_value: float, data: Data
    ) -> tuple[float, np.ndarray]:
        """Calculates gradients."""
        # Calculate loss gradient (dZ)
        dz_loss_gradient = a_activation_value - data.y_true_label

        # Calculate weight gradient array (dW)
        dw_weight_gradients = dz_loss_gradient * data.x_input_vector
        return dz_loss_gradient, dw_weight_gradients

    def update_parameters(
        self, dz_loss_gradient: float, dw_weight_gradients: np.ndarray
    ) -> None:
        """Applies gradients to adjust the neuron's bias and weights."""
        # Update bias
        self.bias = self.bias - (self.learning_rate * dz_loss_gradient)

        # Update weights
        self.weights = self.weights - (self.learning_rate * dw_weight_gradients)
        return
