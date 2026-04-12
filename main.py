from classes.activation_function import ActivationFunction
from classes.algorithms import cross_entropy
from classes.data import Data
from classes.error_calculator import ErrorCalculator
from classes.input_manager import InputManager
from classes.neuron import Neuron

LEARNING_RATE = 0.1
# TODO: connect bias properly
# TODO: Speed-up training & separate from prediction


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
                calculate_loss(activation_value, data)
                print(
                    "Image is:", "Pitch black" if activation_value > 0.5 else "Bright"
                )
                backward_propagation(activation_value, data, neuron)

            case "q":
                exit = True
            case _:
                print("Bad input")
