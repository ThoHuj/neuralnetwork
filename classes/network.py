import numpy as np
from classes.algorithm import Algorithm
from classes.layer import Layer
from classes.neural_network_architecture import NeuralNetworkArchitecture


class Network:
    LEARNING_RATE: float = 0.01
    layers: list[Layer]


    def initialize_layers(
        self, neural_network_architecture: NeuralNetworkArchitecture, seed: int = 99
    ) -> None:

        np.random.seed(seed)
        self.layers = []

        for layer_architecture in neural_network_architecture.layer_structure:
            assert isinstance(layer_architecture["input_dimensions"], int)
            layer_input_size: int = layer_architecture["input_dimensions"]
            assert isinstance(layer_architecture["output_dimensions"], int)
            layer_output_size: int = layer_architecture["output_dimensions"]
            assert isinstance(layer_architecture["activation_function"], str)
            activation_function_key: str = layer_architecture["activation_function"]

            weights = np.random.randn(layer_output_size, layer_input_size) * 0.1
            biases = np.random.randn(layer_output_size, 1) * 0.1

            self.layers.append(Layer(weights, biases, activation_function_key))


    def full_forward_propagation(
        self,
        x_input_vector: np.ndarray,
    ) -> np.ndarray:

        current_a_activation_array = x_input_vector

        for layer in self.layers:
            current_a_activation_array, _ = layer.single_layer_forward_propagation(
                current_a_activation_array
            )

        return current_a_activation_array


    def full_backward_propagation(
        self,
        a_activation_value_array: np.ndarray,
        y_true_label_array: np.ndarray,
    ) -> None:
        """Orchestrates the backward propagation through all layers of the network."""

        previous_da_loss_gradient = Algorithm.cross_entropy_derivative(
            a_activation_value_array, y_true_label_array
        )

        for layer in reversed(self.layers):
            previous_da_loss_gradient, _, _ = layer.single_layer_backward_propagation(
                previous_da_loss_gradient
            )


    def train_model(
        self,
        iterations: int,
        batch_size: int = 10000,
    ) -> list[float]:

        loss_history: list[float] = []

        for iteration in range(iterations):
            print("\rRunning iteration", iteration + 1, "of", iterations, end="")
            random_image_data_vector = np.random.rand(2, batch_size)
            column_means = np.mean(random_image_data_vector, axis=0)
            y_true_label = (column_means < 0.5).astype(float).reshape(1, -1)

            a_activation_array = self.full_forward_propagation(
                random_image_data_vector
            )
            loss = Algorithm.cross_entropy(a_activation_array, y_true_label)
            loss_history.append(loss)

            self.full_backward_propagation(a_activation_array, y_true_label)
            self.update_layers()

        return loss_history


    def update_layers(self) -> None:
        for layer in self.layers:
            layer.current_weights -= (
                self.LEARNING_RATE * layer.current_dw_weight_gradient_array
            )
            layer.current_biases -= (
                self.LEARNING_RATE * layer.current_db_bias_gradient_array
            )
