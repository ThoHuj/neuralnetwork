from classes.activation_function import ActivationFunction
from classes.data import Data
from classes.error_calculator import ErrorCalculator
from classes.neuron import Neuron


def train_model(data: Data):
    for neuron in neurons:
        sensed_data = neuron.sense_data(data_elements=data.vector)
        activation_value = activation_function.process_signal(sensed_data)
        error_delta = error_calculator.calculate_error(
            activation_value, truth_value=data.truth[neuron.name]
        )
        for data_element in data.vector:
            -error_calculator.calculate_debt(
                error_delta,
            )
        neuron.adjust_weights()


if __name__ == "__main__":
    data = Data(
        name="Mörk",
        data_vector=[0.75, 0.8],
        truth=[{"white": 0.0, "gray": 1.0, "black": 0.0}],
    )
    neurons: list[Neuron] = [
        Neuron(name="white", weights=[0.8, 0.5]),
        Neuron(name="gray", weights=[0.1, 0.3]),
        Neuron(name="black", weights=[0.2, 0.4]),
    ]
    activation_function = ActivationFunction()
    error_calculator = ErrorCalculator()
