from math import exp


def cross_entropy(activation_value: float, correct_value: float) -> float:
    return activation_value - correct_value


def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))


def relu(x: float) -> float:
    return max(0, x)
