from classes.activation_function import ActivationFunction
from classes.data import Data
from classes.gradient_calculator import GradientCalculator


class Neuron:
    """A neuron that is sensetive to a specific feature."""

    def __init__(
        self,
        name: str,
        weights: list[float],
        activation_function: ActivationFunction,
        gradient_calculator: GradientCalculator,
        learning_rate: float,
        bias: float = 1.0,
    ):
        self.feature = name
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function
        self.gradient_calculator = gradient_calculator
        self.learning_rate = learning_rate

    def calculate_pre_activation_value(self, data_vector: list[float]) -> float:
        """'Senses' the incoming data based on weights. Returns the pre activation value."""
        activation_vector = [
            data_vector[i] * self.weights[i] for i in range(len(data_vector))
        ]
        pre_activation_value = sum(activation_vector) + self.bias
        return pre_activation_value

    def forward_propagation(
        self,
        data_vector: list[float],
        print_info: bool = False,
    ) -> float:
        if print_info:
            print("Running forward propagation with data vector:", data_vector)
        pre_activation_value = self.calculate_pre_activation_value(data_vector)
        if print_info:
            print("Pre_activation_value:", pre_activation_value)
        activation_value = self.activation_function.process_signal(pre_activation_value)
        if print_info:
            print("Prediction_value:", activation_value)
        return activation_value

    def backward_propagation(
        self, activation_value: float, data: Data, print_info: bool = False
    ) -> None:
        if print_info:
            print("Calculating error gradient")
        error_gradient = self.gradient_calculator.calculate_error_gradient(
            activation_value, label=data.label
        )
        if print_info:
            print(f"Error gradient is: '{error_gradient}'")
        if print_info:
            print("Calculating weight gradients")
        # One element is one gradient for that weight
        gradient_vector = [
            self.gradient_calculator.calculate_weight_gradient(
                error_gradient=error_gradient, data_element=data.vector[i]
            )
            for i in range(len(self.weights))
        ]
        if print_info:
            print(f"Gradients are: '{gradient_vector}'")
        if print_info:
            print(f"Tweaking bias from '{self.bias}'")
        self.bias = self.bias - (self.learning_rate * error_gradient)
        if print_info:
            print(f"Tweaked bias to '{self.bias}'")
        if print_info:
            print(f"Tweaking weights from '{self.weights}'")
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - (
                self.learning_rate * gradient_vector[i]
            )
        if print_info:
            print(f"Tweaked weights to '{self.weights}'")
        return
