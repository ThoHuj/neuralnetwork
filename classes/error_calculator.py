class ErrorCalculator:
    def __init__(self):
        pass

    def cross_entropy(self, activation_value: float, correct_value: float) -> float:
        return activation_value - correct_value

    def calculate_error(self, activation_value: float, truth_value: float) -> float:
        """
        Calculates the error between a node's activation value
        and which value it should have had to be fully correct.
        """
        error_value = self.cross_entropy(activation_value, truth_value)
        return error_value

    def calculate_debt(self, error_value: float, data_element: float) -> float:
        debt = error_value * data_element
        return debt
