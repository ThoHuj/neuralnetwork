import numpy as np


class Algorithm:
    @staticmethod
    def cross_entropy(a_activation_value: float, y_true_label: float) -> float:
        """Calculates the loss for a single data example."""
        # Epsilon is used to clip the activation_value slightly away from 0 and 1
        # so np.log(0) doesn't throw a NaN error.
        epsilon = 1e-15
        a_activation_value_clipped = np.clip(a_activation_value, epsilon, (1 - epsilon))

        # Binary cross-entropy calculation
        loss = -(
            y_true_label * np.log(a_activation_value_clipped)
            + (1 - y_true_label) * np.log(1 - a_activation_value_clipped)
        )
        return float(loss)

    @staticmethod
    def cross_entropy_derivative(
        a_activation_value: float, y_true_label: float
    ) -> float:
        """Calculates the derivative of the loss (dA) for a single data example."""
        epsilon = 1e-15
        a_activation_value_clipped = np.clip(a_activation_value, epsilon, (1 - epsilon))

        # Binary cross-entropy derivative calculation
        da_cross_entropy_derivative = -(y_true_label / a_activation_value_clipped) + (
            (1 - y_true_label) / (1 - a_activation_value_clipped)
        )
        return float(da_cross_entropy_derivative)

    @staticmethod
    def sigmoid(z_pre_activation_value: np.ndarray) -> np.ndarray:
        """Using sigmoid to calculate activation value (a) from pre activation value (z)."""
        # Clipping to avoid overflow warnings for large numbers
        clipped_z_pre_activation_value = np.clip(z_pre_activation_value, -500, 500)
        a_activation_value = 1 / (1 + np.exp(-clipped_z_pre_activation_value))
        return a_activation_value

    @staticmethod
    def sigmoid_derivative(z_pre_activation_value: np.ndarray) -> np.ndarray:
        """Using sigmoid to calculate g prime value from the provided pre activation value (z)."""
        sigmoid_product = Algorithm.sigmoid(z_pre_activation_value)
        g_prime_value = sigmoid_product * (1 - sigmoid_product)
        return g_prime_value

    @staticmethod
    def relu(z_pre_activation_value: np.ndarray) -> np.ndarray:
        """Using ReLU to calculate activation value (a) from pre activation value (z)."""
        a_activation_value = np.max(0.0, z_pre_activation_value)
        return a_activation_value

    @staticmethod
    def relu_derivative(z_pre_activation_value: np.ndarray) -> np.ndarray:
        """Using ReLU to calculate g prime value from the provided pre activation value (z)."""
        g_prime_value = z_pre_activation_value > 0
        return g_prime_value
