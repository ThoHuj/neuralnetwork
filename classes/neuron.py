class Neuron:
    """A neuron that is sensetive to a specific feature."""

    weights: list[float]
    feature: str
    bias: float

    def __init__(self, name: str, weights: list[float], bias: float = 1.0):
        self.feature = name
        self.weights = weights
        self.bias = bias

    def calculate_pre_activation_value(self, data_vector: list[float]) -> float:
        """'Senses' the incoming data based on weights. Returns the pre activation value."""
        activation_vector = [
            data_vector[i] * self.weights[i] for i in range(len(data_vector))
        ]
        pre_activation_value = sum(activation_vector) + self.bias
        return pre_activation_value
