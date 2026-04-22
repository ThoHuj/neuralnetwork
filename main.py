from random import random, uniform

import matplotlib

from classes.gradient_calculator import GradientCalculator
from classes.loss_calculator import LossCalculator

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotx  # type: ignore
import numpy as np

from classes.activation_function import ActivationFunction
from classes.algorithms import Algorithms
from classes.data import Data
from classes.input_manager import InputManager
from classes.neuron import Neuron

plt.style.use(matplotx.styles.pitaya_smoothie["dark"])  # type: ignore

LEARNING_RATE = 10.0


def randomize_dark_image_data() -> list[float]:
    dark_data_x = uniform(a=0.95, b=1.0)
    dark_data_y = uniform(a=0.95, b=1.0)
    return [dark_data_x, dark_data_y]


def randomize_bright_image_data() -> list[float]:
    bright_data_x = uniform(a=0.0, b=0.95)
    bright_data_y = uniform(a=0.0, b=0.95)
    return [bright_data_x, bright_data_y]


def train_model(
    neuron: Neuron, loss_calculator: LossCalculator, print_info: bool = True
) -> list[float]:
    iterations = input_manager.prompt_for_integer(
        prompt="Enter a number of data sets to train with: "
    )
    loss_history: list[float] = []
    for iteration in range(iterations):
        # Generate and construct data
        random_image_values = (
            randomize_dark_image_data()
            if iteration % 2 == 0
            else randomize_bright_image_data()
        )
        image_data = np.array(random_image_values)
        data = Data(label=1.0 if iteration % 2 == 0 else 0.0, vector=image_data)

        # Execute forward propagation
        pre_activation_value, activation_value = neuron.forward_propagation(data.vector)

        # Calculate loss
        loss = Algorithms.cross_entropy(activation_value, y_true_label=data.label)
        loss_history.append(loss)

        # Execute backward propagation
        neuron.backward_propagation(activation_value, data)
        if print_info:
            print(f"\rIteration: {iteration + 1} of {iterations}", end="", flush=True)
    return loss_history


def print_prediction(activation_value: float) -> None:
    print("Image is:", "Pitch black" if activation_value > 0.5 else "Bright")


def plot_loss_history(loss_history: list[float]) -> None:
    reduced_loss_history = loss_history[::100]
    plt.plot(reduced_loss_history, marker="o")  # type: ignore
    plt.show()  # type: ignore


# Pre trained values:
# weights=[78.02563146707283, 77.20075351896011],  # [random(), random()]
# bias=-143.73853640727853,

if __name__ == "__main__":
    activation_function = ActivationFunction()
    gradient_calculator = GradientCalculator()
    loss_calculator = LossCalculator()
    neuron = Neuron(
        name="white",
        weights=[random(), random()],
        bias=random(),
        activation_function=activation_function,
        gradient_calculator=gradient_calculator,
        learning_rate=LEARNING_RATE,
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
                activation_value = neuron.forward_propagation(
                    x_input_vector=data.vector
                )
                print_prediction(activation_value)
            case "2":
                loss_history += train_model(neuron, loss_calculator)
                plot_loss_history(loss_history)
            case "q":
                exit = True
            case _:
                print("Bad input")
