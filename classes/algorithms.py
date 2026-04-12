from math import exp, log


def cross_entropy(activation_value: float, correct_value: float) -> float:
    # Prevents math domain errors (since a log value of 0 crashes Python)
    epsilon = 1e-15
    a = max(epsilon, min(1 - epsilon, activation_value))

    # Binary cross-entropy calculation
    loss = -(correct_value * log(a) + (1 - correct_value) * log(1 - a))
    return loss


def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))


def relu(x: float) -> float:
    return max(0, x)
