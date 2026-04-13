from classes.algorithms import cross_entropy


class LossCalculator:
    def calculate_loss(
        self, activation_value: float, correct_value: float, print_info: bool = False
    ) -> float:
        if print_info:
            print("Calculating loss")
        loss = cross_entropy(activation_value, correct_value)
        if print_info:
            print("Loss:", loss)
        return loss
