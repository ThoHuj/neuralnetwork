import math


class ActivationFunction:
    def __init__(self):
        pass

    def sigmoid(self, x: float):
        return 1 / (1 + math.exp(-x))

    def relu(self, x: float) -> float:
        return max(0, x)

    def process_signal(self, sense_value: float) -> float:
        """
        Takes an incoming sense value (from a neuron) and process it through
        an agorithm to determine its activation value.
        """
        activation_value = self.sigmoid(
            sense_value
        )  # Change function to try different activation functions.
        return activation_value
