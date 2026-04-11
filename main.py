from classes.activation_function import ActivationFunction
from classes.algorithms import cross_entropy
from classes.data import Data
from classes.error_calculator import ErrorCalculator
from classes.input_manager import InputManager
from classes.neuron import Neuron

LEARNING_RATE = 0.5


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
    loss = cross_entropy(activation_value, correct_value=data.label) ** 2
    print("Loss:", loss)
    return loss


def calculate_gradient(
    activation_value: float, label: float, data_element: float
) -> float:
    gradient = 2 * (activation_value - label) * data_element
    return gradient


def backward_propagation(activation_value: float, data: Data, neuron: Neuron):
    # One element is one gradient for that weight
    print("Calculating gradients")
    gradient_vector = [
        calculate_gradient(activation_value, data.label, data.vector[i])
        for i in range(len(neuron.weights))
    ]
    print("Tweaking weights")
    for i in range(len(neuron.weights)):
        neuron.weights[i] = neuron.weights[i] - (LEARNING_RATE * gradient_vector[i])
    return


if __name__ == "__main__":
    neuron = Neuron(name="white", weights=[0.5, 0.5])
    activation_function = ActivationFunction()
    error_calculator = ErrorCalculator()
    input_manager = InputManager()

    exit = False
    while exit is False:
        input_value = input("1 - Predict\n2 - Train\nq - quit\nChoice: ")
        match input_value:
            case "1":
                data = input_manager.prompt_for_data()
                activation_value = forward_propagation(
                    data.vector, neuron, activation_function
                )
                backward_propagation(activation_value, data, neuron)

            case "q":
                exit = True
            case _:
                print("Bad input")
