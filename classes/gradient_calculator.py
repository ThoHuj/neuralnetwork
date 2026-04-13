class GradientCalculator:
    def calculate_error_gradient(self, activation_value: float, label: float) -> float:
        error_gradient = activation_value - label
        return error_gradient

    def calculate_weight_gradient(
        self,
        data_element: float,
        error_gradient: float,
    ) -> float:
        weight_gradient = (error_gradient) * data_element
        return weight_gradient
