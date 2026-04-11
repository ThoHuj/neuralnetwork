from classes.algorithms import cross_entropy


class ErrorCalculator:
    def calculate_error(self, activation_value: float, truth_value: float) -> float:
        """
        Calculates the error between a node's activation value
        and which value it should have had to be fully correct.
        """
        error_value = cross_entropy(activation_value, truth_value)
        return error_value

    def calculate_debt(self, error_value: float, weight: float) -> float:
        debt = error_value * weight
        return debt
