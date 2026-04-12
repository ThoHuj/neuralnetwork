from classes.activation_function import ActivationFunction


class Neuron:
    """A neuron that is sensetive to a specific feature."""

    weights: list[float]
    feature: str
    bias: float
    activation_function: ActivationFunction

    def __init__(
        self,
        name: str,
        weights: list[float],
        activation_function: ActivationFunction,
        bias: float = 1.0,
    ):
        self.feature = name
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function

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
