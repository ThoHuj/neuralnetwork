from random import random, uniform

import matplotlib

matplotlib.use("QtAgg")
from typing import Callable

import matplotlib.pyplot as plt
import matplotx  # type: ignore
import numpy as np

from classes.algorithm import Algorithm
from classes.data import Data
from classes.input_manager import InputManager
from classes.neuron import Neuron

plt.style.use(matplotx.styles.pitaya_smoothie["dark"])  # type: ignore

LEARNING_RATE = 0.01

NEURAL_NETWORK_ARCHITECTURE: list[dict[str, int | str]] = [
    {"input_dimensions": 2, "output_dimensions": 25, "activation_function": "relu"},
    {"input_dimensions": 25, "output_dimensions": 50, "activation_function": "relu"},
    {"input_dimensions": 50, "output_dimensions": 50, "activation_function": "relu"},
    {"input_dimensions": 50, "output_dimensions": 1, "activation_function": "sigmoid"},
]


def initialize_layers(
    neural_network_architecture: list[dict[str, int | str]], seed: int = 99
):
    np.random.seed(seed)
    state_dict: dict[str, np.ndarray] = {}

    for index, layer in enumerate(neural_network_architecture):
        # Just to prevent layer index from starting from 0
        layer_index = index + 1

        # Extract number of dimensions from neural network architecture
        assert type(layer["input_dimensions"]) is int
        layer_input_size: int = layer["input_dimensions"]
        assert type(layer["output_dimensions"]) is int
        layer_output_size: int = layer["output_dimensions"]

        # Create a weight matrix and seed with random values
        state_dict["W" + str(layer_index)] = (
            np.random.randn(layer_output_size, layer_input_size) * 0.1
        )
        # Create a bias matrix and seed with random values
        state_dict["b" + str(layer_index)] = np.random.randn(layer_output_size, 1) * 0.1
    return state_dict


def single_layer_forward_propagation(
    previous_layer_a_activation: np.ndarray,
    current_w_weights: np.ndarray,
    current_b_biases: np.ndarray,
    activation_function_key: str = "relu",
) -> tuple[np.ndarray, np.ndarray]:
    # Calculate pre activation values (z)
    z_pre_activation_values = (
        np.dot(current_w_weights, previous_layer_a_activation) + current_b_biases
    )

    # Set activation function
    match activation_function_key:
        case "relu":
            activation_function: Callable[[np.ndarray], np.ndarray] = Algorithm.relu
        case "sigmoid":
            activation_function: Callable[[np.ndarray], np.ndarray] = Algorithm.sigmoid
        case _:
            raise RuntimeError("Provided function does not exist.")

    # Use activation function to produce activation_data
    # Also includes the pre activation values.
    a_activation_data: tuple[np.ndarray, np.ndarray] = (
        activation_function(z_pre_activation_values),
        z_pre_activation_values,
    )
    return a_activation_data


def full_forward_propagation(
    x_input_vector: np.ndarray,
    state_dict: dict[str, np.ndarray],
    neural_network_architecture: list[dict[str, int | str]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    memory: dict[str, np.ndarray] = {}
    current_a_activation_array = x_input_vector

    for index, layer in enumerate(neural_network_architecture):
        layer_index = index + 1
        previous_a_activation_array = current_a_activation_array

        assert type(layer["activation_function"]) is str
        activation_function = layer["activation_function"]
        current_w_weight_array = state_dict["W" + str(layer_index)]
        current_b_bias_array = state_dict["b" + str(layer_index)]

        current_a_activation_array, z_pre_activation_array = (
            single_layer_forward_propagation(
                previous_a_activation_array,
                current_w_weight_array,
                current_b_bias_array,
                activation_function,
            )
        )

        memory["A" + str(index)] = previous_a_activation_array
        memory["Z" + str(layer_index)] = z_pre_activation_array

    return current_a_activation_array, memory


def randomize_dark_image_data() -> list[float]:
    dark_data_x = uniform(a=0.25, b=0.0)
    dark_data_y = uniform(a=0.25, b=0.0)
    return [dark_data_x, dark_data_y]


def randomize_bright_image_data() -> list[float]:
    bright_data_x = uniform(a=0.25, b=1.00)
    bright_data_y = uniform(a=0.25, b=1.00)
    return [bright_data_x, bright_data_y]


def train_model(
    neuron: Neuron, iterations: int, print_info: bool = True
) -> list[float]:
    loss_history: list[float] = []

    for iteration in range(iterations):
        # Generate and construct input vector with true labels
        random_image_data = (
            randomize_dark_image_data()
            if iteration % 2 == 0
            else randomize_bright_image_data()
        )
        image_data_array = np.array(random_image_data)
        data = Data(
            y_true_label=1.0 if iteration % 2 == 0 else 0.0,
            x_input_vector=image_data_array,
        )

        # Execute forward propagation on neuron with generated data
        z_pre_activation_value, a_activation_value = neuron.forward_propagation(
            data.x_input_vector
        )

        # Calculate loss
        loss = Algorithm.cross_entropy(
            a_activation_value, y_true_label=data.y_true_label
        )
        loss_history.append(loss)

        # Execute backward propagation to generate gradients
        dz_loss_gradient, dw_weights_gradient = neuron.backward_propagation(
            a_activation_value, data
        )

        # Tweak weights and bias
        neuron.update_parameters(dz_loss_gradient, dw_weights_gradient)

        # Print info
        if print_info:
            print(f"\rIteration: {iteration + 1} of {iterations}", end="", flush=True)
    return loss_history


def print_prediction(activation_value: float) -> None:
    print("Image is:", "Dark" if activation_value > 0.5 else "Bright")


def plot_loss_history(loss_history: list[float]) -> None:
    reduced_loss_history = loss_history[::100]
    plt.plot(reduced_loss_history, marker="o")  # type: ignore
    plt.show()  # type: ignore


def main() -> None:
    neuron = Neuron(
        name="white",
        weights=np.array([random(), random()]),
        bias=random(),
        learning_rate=LEARNING_RATE,
        activation_function=Algorithm.sigmoid,
    )
    input_manager = InputManager()
    loss_history: list[float] = []
    exit = False

    while exit is False:
        menu_choice = input_manager.prompt_for_string(
            "1 - Predict\n2 - Train\nq - quit\nChoice: "
        )
        match menu_choice:
            case "1":
                data = input_manager.prompt_for_data(enter_label=False)
                z_pre_activation_value, a_activation_value = neuron.forward_propagation(
                    x_input_vector=data.x_input_vector
                )
                print_prediction(a_activation_value)
            case "2":
                iterations = input_manager.prompt_for_integer(
                    prompt="Enter a number of data sets to train with: "
                )
                loss_history += train_model(neuron, iterations)
                plot_loss_history(loss_history)
            case "q":
                exit = True
            case _:
                print("Bad input")


if __name__ == "__main__":
    main()
