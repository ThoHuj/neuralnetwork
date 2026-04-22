from random import random, uniform

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotx  # type: ignore
import numpy as np

from classes.algorithm import Algorithm
from classes.data import Data
from classes.input_manager import InputManager
from classes.neuron import Neuron

plt.style.use(matplotx.styles.pitaya_smoothie["dark"])  # type: ignore

LEARNING_RATE = 0.0001


def randomize_dark_image_data() -> list[float]:
    dark_data_x = uniform(a=0.95, b=1.0)
    dark_data_y = uniform(a=0.95, b=1.0)
    return [dark_data_x, dark_data_y]


def randomize_bright_image_data() -> list[float]:
    bright_data_x = uniform(a=0.0, b=0.95)
    bright_data_y = uniform(a=0.0, b=0.95)
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
    print("Image is:", "Pitch black" if activation_value > 0.5 else "Bright")


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
