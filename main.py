import matplotlib

matplotlib.use("QtAgg")
from typing import Callable

import matplotlib.pyplot as plt
import matplotx  # type: ignore
import numpy as np

from classes.algorithm import Algorithm
from classes.data import Data
from classes.input_manager import InputManager

plt.style.use(matplotx.styles.pitaya_smoothie["dark"])  # type: ignore

LEARNING_RATE = 0.01

NEURAL_NETWORK_ARCHITECTURE: list[dict[str, int | str]] = [
    {"input_dimensions": 2, "output_dimensions": 25, "activation_function": "relu"},
    {"input_dimensions": 25, "output_dimensions": 50, "activation_function": "relu"},
    {"input_dimensions": 50, "output_dimensions": 50, "activation_function": "relu"},
    {"input_dimensions": 50, "output_dimensions": 1, "activation_function": "sigmoid"},
]

# TODO: There is a more modern way of type annotating ndarrays


def initialize_layers(
    neural_network_architecture: list[dict[str, int | str]], seed: int = 99
) -> dict[str, np.ndarray]:

    np.random.seed(seed)
    state_dict: dict[str, np.ndarray] = {}

    for index, layer in enumerate(neural_network_architecture):
        # Just to prevent layer index from starting from 0
        layer_index = index + 1

        # Extract number of dimensions from neural network architecture
        assert isinstance(layer["input_dimensions"], int)
        layer_input_size: int = layer["input_dimensions"]
        assert isinstance(layer["output_dimensions"], int)
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
            g_activation_function: Callable[[np.ndarray], np.ndarray] = Algorithm.relu
        case "sigmoid":
            g_activation_function: Callable[[np.ndarray], np.ndarray] = (
                Algorithm.sigmoid
            )
        case _:
            raise RuntimeError("Requested activation function does not exist.")

    # Use activation function to produce activation_data
    # Also includes the pre activation values.
    a_activation_data: tuple[np.ndarray, np.ndarray] = (
        g_activation_function(z_pre_activation_values),
        z_pre_activation_values,
    )
    return a_activation_data


def full_forward_propagation(
    x_input_vector: np.ndarray,
    state_dict: dict[str, np.ndarray],
    neural_network_architecture: list[dict[str, int | str]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:

    forward_propagation_cache: dict[str, np.ndarray] = {}
    current_a_activation_array = x_input_vector

    for index, layer in enumerate(neural_network_architecture):
        layer_index = index + 1
        previous_a_activation_array = current_a_activation_array

        assert isinstance(layer["activation_function"], str)
        layer_activation_function_key = layer["activation_function"]
        current_w_weight_array = state_dict["W" + str(layer_index)]
        current_b_bias_array = state_dict["b" + str(layer_index)]

        current_a_activation_array, z_pre_activation_array = (
            single_layer_forward_propagation(
                previous_a_activation_array,
                current_w_weight_array,
                current_b_bias_array,
                layer_activation_function_key,
            )
        )

        forward_propagation_cache["A" + str(index)] = previous_a_activation_array
        forward_propagation_cache["Z" + str(layer_index)] = z_pre_activation_array

        # print(current_a_activation_array.shape[1])

    final_a_activation_array = current_a_activation_array
    # print(
    #     "Fullforward is done. final activation array length is",
    #     final_a_activation_array.shape[1],
    # )
    return final_a_activation_array, forward_propagation_cache


def single_layer_backward_propagation(
    current_da_loss_gradient: np.ndarray,
    current_w_weight_array: np.ndarray,
    current_z_pre_activation_array: np.ndarray,
    previous_a_activation_array: np.ndarray,
    activation_function_key: str = "relu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m_batch_size = previous_a_activation_array.shape[1]

    # Set activation function (G)
    match activation_function_key:
        case "relu":
            g_activation_function_derivative: Callable[
                [np.ndarray, np.ndarray], np.ndarray
            ] = Algorithm.relu_derivative
        case "sigmoid":
            g_activation_function_derivative: Callable[
                [np.ndarray, np.ndarray], np.ndarray
            ] = Algorithm.sigmoid_derivative
        case _:
            raise RuntimeError("Requested activation function does not exist.")

    # Calculate this layer's pre activation gradient (dZ)
    current_dz_pre_activation_gradient = g_activation_function_derivative(
        current_da_loss_gradient, current_z_pre_activation_array
    )

    # Calculate this layer's weight gradients (dW)
    current_dw_weight_gradient_array = (
        np.dot(current_dz_pre_activation_gradient, previous_a_activation_array.T)
        / m_batch_size
    )

    # Calculate this layer's bias gradient (dB)
    current_db_bias_gradient_array = (
        np.sum(current_dz_pre_activation_gradient, axis=1, keepdims=True) / m_batch_size
    )

    # Calculate loss gradient (dA) before changing weights
    previous_da_loss_gradient = np.dot(
        current_w_weight_array.T, current_dz_pre_activation_gradient
    )

    return (
        previous_da_loss_gradient,
        current_dw_weight_gradient_array,
        current_db_bias_gradient_array,
    )


def full_backward_propagation(
    a_activation_value_array: np.ndarray,
    y_true_label_array: np.ndarray,
    forward_propagation_cache: dict[str, np.ndarray],
    state_dict: dict[str, np.ndarray],
    neural_network_architecture: list[dict[str, int | str]],
) -> dict[str, np.ndarray]:
    """Orchestrates the backward propagation through all layers of the network."""

    gradients_dict: dict[str, np.ndarray] = {}

    # Calculate loss gradient (dA)
    previous_da_loss_gradient = Algorithm.cross_entropy_derivative(
        a_activation_value_array, y_true_label_array
    )

    # Loop backwards through all layers in network
    for index, layer_configuration in reversed(
        list(enumerate(neural_network_architecture))
    ):
        # Layers start at 1, not 0
        layer_index = index + 1

        # Fetch this specific layer's activation function key
        assert isinstance(layer_configuration["activation_function"], str)
        layer_activation_function_key: str = layer_configuration["activation_function"]

        # The loss gradient (dA) that was outputted from the previous loop
        # becomes the input loss gradient for the current layer
        current_da_loss_gradient = previous_da_loss_gradient

        # Obtain the cached data for this specific layer
        previous_a_activation_array = forward_propagation_cache["A" + str(index)]
        current_z_pre_activation_array = forward_propagation_cache[
            "Z" + str(layer_index)
        ]

        # Obtain the weights for this specific layer
        current_w_weight_array = state_dict["W" + str(layer_index)]

        # Calculate gradients for the current layer and get the loss gradient (dA) for the next layer
        (
            previous_da_loss_gradient,
            current_dw_weight_gradient_array,
            current_db_bias_gradient_array,
        ) = single_layer_backward_propagation(
            current_da_loss_gradient,
            current_w_weight_array,
            current_z_pre_activation_array,
            previous_a_activation_array,
            layer_activation_function_key,
        )

        # Store gradients for updating weights later
        gradients_dict["dW" + str(layer_index)] = current_dw_weight_gradient_array
        gradients_dict["db" + str(layer_index)] = current_db_bias_gradient_array

    return gradients_dict


def update_layers(
    state_dict: dict[str, np.ndarray],
    gradients_dict: dict[str, np.ndarray],
    neural_network_architecture: list[dict[str, int | str]],
    learning_rate: float,
):
    # Iterate over each layer and subtract their corresponding weight gradient (dW) and bias gradient (db).
    # This modifies the weights and the bias of all neurons in a layer.
    for index, _ in enumerate(neural_network_architecture):
        layer_index = index + 1

        state_dict["W" + str(layer_index)] -= (
            learning_rate * gradients_dict["dW" + str(layer_index)]
        )
        state_dict["b" + str(layer_index)] -= (
            learning_rate * gradients_dict["db" + str(layer_index)]
        )

    return state_dict


def train_model(
    iterations: int, neural_network_state_dict: dict[str, np.ndarray]
) -> list[float]:

    loss_history: list[float] = []

    for iteration in range(iterations):
        print("\rRunning iteration", iteration + 1, "of", iterations)
        random_image_data_vector = np.random.rand(1, 2)
        y_true_label = (
            np.array([[1.0]])
            if np.mean(random_image_data_vector) < 0.5
            else np.array([[0.0]])
        )
        generated_image = Data(y_true_label, random_image_data_vector)
        a_activation_array, forward_propagation_cache = full_forward_propagation(
            generated_image.x_input_vector,
            neural_network_state_dict,
            NEURAL_NETWORK_ARCHITECTURE,
        )
        loss = Algorithm.cross_entropy(a_activation_array, generated_image.y_true_label)
        loss_history.append(loss)

        gradients_dict: dict[str, np.ndarray] = full_backward_propagation(
            a_activation_array,
            y_true_label,
            forward_propagation_cache,
            neural_network_state_dict,
            NEURAL_NETWORK_ARCHITECTURE,
        )
        new_state_dict = update_layers(
            neural_network_state_dict,
            gradients_dict,
            NEURAL_NETWORK_ARCHITECTURE,
            LEARNING_RATE,
        )

    return loss_history


def plot_loss_history(loss_history: list[float]) -> None:
    reduced_loss_history = loss_history[::100]
    plt.plot(reduced_loss_history, marker="o")  # type: ignore
    plt.show()  # type: ignore


def main() -> None:
    neural_network_state_dict = initialize_layers(NEURAL_NETWORK_ARCHITECTURE)
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
                a_activation_array, _ = full_forward_propagation(
                    data.x_input_vector,
                    neural_network_state_dict,
                    NEURAL_NETWORK_ARCHITECTURE,
                )
                print(a_activation_array.shape[1])
                print("Predicions: ")
                for index, probability in enumerate(a_activation_array.flatten()):
                    predicted_class = "Bright" if probability < 0.5 else "Dark"
                    print(
                        "Output number: ",
                        index,
                        "\nPrediction: ",
                        predicted_class,
                        "\nProbability:",
                        probability,
                    )

            case "2":
                iterations = input_manager.prompt_for_integer(
                    prompt="Enter a number of data sets to train with: "
                )

                loss_history += train_model(iterations, neural_network_state_dict)
                plot_loss_history(loss_history)
            case "q":
                exit = True
            case _:
                print("Bad input")


if __name__ == "__main__":
    main()
