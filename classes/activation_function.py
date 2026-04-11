from classes.algorithms import sigmoid


class ActivationFunction:
    def process_signal(self, activation_value: float) -> float:
        """
        Takes an incoming sense value (from a neuron) and process it through
        an agorithm to determine its activation value.
        """
        prediction_value = sigmoid(activation_value)
        return prediction_value
