import numpy as np
from typing import Callable
from classes.algorithm import Algorithm


class Layer:
    input_size: int
    output_size: int
    current_weights: np.ndarray
    current_biases: np.ndarray
    activation_function: Callable[[np.ndarray], np.ndarray]
    activation_function_derivative: Callable[[np.ndarray, np.ndarray], np.ndarray]
    current_dz_pre_activation_gradient: np.ndarray
    current_dw_weight_gradient_array: np.ndarray
    current_db_bias_gradient_array: np.ndarray
    previous_a_activation_array: np.ndarray
    current_z_pre_activation_array: np.ndarray

    def __init__(
        self, weights: np.ndarray, biases: np.ndarray, activation_function_key: str
    ) -> None:
        self.current_weights = weights
        self.current_biases = biases

        # Set activation functions
        match activation_function_key:
            case "relu":
                self.activation_function = Algorithm.relu
                self.activation_function_derivative = Algorithm.relu_derivative
            case "sigmoid":
                self.activation_function = Algorithm.sigmoid
                self.activation_function_derivative = Algorithm.sigmoid_derivative
            case _:
                raise RuntimeError("Requested activation function does not exist.")

    def single_layer_forward_propagation(
        self,
        previous_layer_a_activation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        
        self.previous_a_activation_array = previous_layer_a_activation

        # Calculate pre activation values (z)
        self.current_z_pre_activation_array = (
            np.dot(self.current_weights, previous_layer_a_activation)
            + self.current_biases
        )

        # Use activation function to produce activation_data
        # Also includes the pre activation values.
        a_activation_data: tuple[np.ndarray, np.ndarray] = (
            self.activation_function(self.current_z_pre_activation_array),
            self.current_z_pre_activation_array,
        )
        return a_activation_data

    def single_layer_backward_propagation(
        self,
        current_da_loss_gradient: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m_batch_size = self.previous_a_activation_array.shape[1]

        # Calculate this layer's pre activation gradient (dZ)
        self.current_dz_pre_activation_gradient = self.activation_function_derivative(
            current_da_loss_gradient, self.current_z_pre_activation_array
        )

        # Calculate this layer's weight gradients (dW)
        self.current_dw_weight_gradient_array = (
            np.dot(
                self.current_dz_pre_activation_gradient, self.previous_a_activation_array.T
            )
            / m_batch_size
        )

        # Calculate this layer's bias gradient (dB)
        self.current_db_bias_gradient_array = (
            np.sum(self.current_dz_pre_activation_gradient, axis=1, keepdims=True)
            / m_batch_size
        )

        # Calculate loss gradient (dA) before changing weights
        previous_da_loss_gradient = np.dot(
            self.current_weights.T, self.current_dz_pre_activation_gradient
        )

        return (
            previous_da_loss_gradient,
            self.current_dw_weight_gradient_array,
            self.current_db_bias_gradient_array,
        )
