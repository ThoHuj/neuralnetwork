# GRAYY
class Neuron:
    """A neuron that is sensetive to a specific feature."""

    weight_elements: list[float]
    name: str

    def __init__(self, name: str, weights: list[float]):
        self.name = name
        self.weight_elements = weights

    def sense_data(self, data_elements: list[float]) -> float:
        """Senses the incoming data based on weights. Returns the sense strength."""
        sensed_data_elements = [
            data_elements[i] * self.weight_elements[i]
            for i in range(len(data_elements))
        ]
        sense_sum = sum(sensed_data_elements)
        return sense_sum

    def adjust_weights(self, debt: float, correction_factor: float) -> None:
        new_weights: list[float] = []
        for weight in self.weight_elements:
            new_weight = weight - (correction_factor * debt)
            new_weights.append(new_weight)
        self.weight_elements = new_weights
