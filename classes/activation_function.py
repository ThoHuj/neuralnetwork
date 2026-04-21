from classes.algorithms import sigmoid


class ActivationFunction:
    pre_activation_value = float

    def __init__(self, pre_activation_value: float, function: Algorithm) -> None:
        self.pre_activation_value = pre_activation_value


class Sigmoid(ActivationFunction):
    def process_signal(self, activation_value: float) -> float:
        """
        Takes an incoming sense value (from a neuron) and process it through
        an algorithm to determine its activation value.
        """
        prediction_value = sigmoid(activation_value)
        return prediction_value
