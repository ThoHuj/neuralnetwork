from random import uniform

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotx  # type: ignore

from classes.activation_function import ActivationFunction
from classes.algorithms import cross_entropy
from classes.data import Data
from classes.error_calculator import ErrorCalculator
from classes.input_manager import InputManager
from classes.neuron import Neuron

plt.style.use(matplotx.styles.pitaya_smoothie["dark"])  # type: ignore

LEARNING_RATE = 1.0


def forward_propagation(
    data_vector: list[float], neuron: Neuron, activation_function: ActivationFunction
) -> float:
    print("Running forward propagation with data vector:", data_vector)
    activation_function = activation_function
    pre_activation_value = neuron.calculate_pre_activation_value(data_vector)
    print("Pre_activation_value:", pre_activation_value)
    prediction_value = activation_function.process_signal(pre_activation_value)
    print("Prediction_value:", prediction_value)
    return prediction_value


def calculate_loss(activation_value: float, data: Data) -> float:
    print("Calculating loss")
    loss = cross_entropy(activation_value, correct_value=data.label)
    print("Loss:", loss)
    return loss


def calculate_weight_gradient(
    data_element: float,
    error_gradient: float,
) -> float:
    weight_gradient = (error_gradient) * data_element
    return weight_gradient


def calculate_error_gradient(activation_value: float, label: float) -> float:
    error_gradient = activation_value - label
    return error_gradient


def backward_propagation(activation_value: float, data: Data, neuron: Neuron) -> None:
    print("Calculating error gradient")
    error_gradient = calculate_error_gradient(activation_value, label=data.label)
    print(f"Error gradient is: '{error_gradient}'")
    # One element is one gradient for that weight
    print("Calculating weight gradients")
    gradient_vector = [
        calculate_weight_gradient(
            error_gradient=error_gradient, data_element=data.vector[i]
        )
        for i in range(len(neuron.weights))
    ]
    print(f"Gradients are: '{gradient_vector}'")
    print(f"Tweaking bias from '{neuron.bias}'")
    neuron.bias = neuron.bias - (LEARNING_RATE * error_gradient)
    print(f"Tweaked bias to '{neuron.bias}'")
    print(f"Tweaking weights from '{neuron.weights}'")
    for i in range(len(neuron.weights)):
        neuron.weights[i] = neuron.weights[i] - (LEARNING_RATE * gradient_vector[i])
    print(f"Tweaked weights to '{neuron.weights}'")
    return


def randomize_dark_image_data() -> list[float]:
    dark_data_x = uniform(a=0.95, b=1.0)
    dark_data_y = uniform(a=0.95, b=1.0)
    return [dark_data_x, dark_data_y]


def randomize_bright_image_data() -> list[float]:
    bright_data_x = uniform(a=0.0, b=0.95)
    bright_data_y = uniform(a=0.0, b=0.95)
    return [bright_data_x, bright_data_y]


def train_model() -> list[float]:
    iterations = input_manager.prompt_for_integer(
        prompt="Enter a number of data sets to train with: "
    )
    loss_history: list[float] = []
    for iteration in range(iterations):
        image_data = (
            randomize_dark_image_data()
            if iteration % 2 == 0
            else randomize_bright_image_data()
        )
        data = Data(label=1.0 if iteration % 2 == 0 else 0.0, vector=image_data)
        activation_value = forward_propagation(data.vector, neuron, activation_function)
        loss = calculate_loss(activation_value, data)
        loss_history.append(loss)
        backward_propagation(activation_value, data, neuron)
        print("Image is:", "Pitch black" if activation_value > 0.5 else "Bright     ")
        print(f"\rIteration: {iteration + 1} of {iterations}", end="", flush=True)
    return loss_history


if __name__ == "__main__":
    # random(), random()
    neuron = Neuron(
        name="white",
        weights=[78.02563146707283, 77.20075351896011],
        bias=-143.73853640727853,
    )
    activation_function = ActivationFunction()
    error_calculator = ErrorCalculator()
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
                activation_value = forward_propagation(
                    data.vector, neuron, activation_function
                )
                """ calculate_loss(activation_value, data)"""
                print(
                    "Image is:", "Pitch black" if activation_value > 0.5 else "Bright"
                )
                """backward_propagation(activation_value, data, neuron) """
            case "2":
                loss_history += train_model()
                reduced_loss_history = loss_history[::100]
                plt.plot(reduced_loss_history, marker="o")  # type: ignore
                plt.show()  # type: ignore
            case "q":
                exit = True
            case _:
                print("Bad input")
