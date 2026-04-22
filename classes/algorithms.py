from typing import Union

import numpy as np


class Algorithms:
    @staticmethod
    def cross_entropy(activation_value: np.ndarray, correct_value: np.ndarray) -> float:
        # Epsilon is a tiny number to prevent taking the log of exactly 0 or 1
        # This is used because a log value of 0 would crash Python
        epsilon = 1e-15

        # Epsilon is used to clip the activation_value slightly away from 0 and 1
        # so np.log(0) doesn't throw a NaN (Not a Number) error.
        activation_value_clipped = np.clip(activation_value, epsilon, (1 - epsilon))

        # Binary cross-entropy calculation
        loss = -np.mean(
            correct_value * np.log(activation_value_clipped)
            + (1 - correct_value) * np.log(1 - activation_value_clipped)
        )
        return loss.item()  # Extracts the float from the np object

    @staticmethod
    def cross_entropy_derivative(
        activation_value_array: np.ndarray, correct_value_array: np.ndarray
    ) -> np.ndarray:
        epsilon = 1e-15
        activation_value_array_clipped = np.clip(
            activation_value_array, epsilon, (1 - epsilon)
        )

        # Binary cross-entropy derivative calculation
        cross_entropy_derivative = -(
            np.divide(correct_value_array, activation_value_array_clipped)
            - np.divide(1 - correct_value_array, 1 - activation_value_array_clipped)
        )
        # Normalize by batch size since the forward pass uses np.mean()
        normalized_cross_entropy_derivative = cross_entropy_derivative / np.size(
            correct_value_array
        )
        return normalized_cross_entropy_derivative

    @staticmethod
    def sigmoid(
        pre_activation_value: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        # Clipping to avoid overflow warnings for large numbers
        clipped_pre_activation_value = np.clip(pre_activation_value, -500, 500)
        return 1 / (1 + np.exp(-clipped_pre_activation_value))

    @staticmethod
    def sigmoid_derivative(
        pre_activation_values: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        sigmoid_product = Algorithms.sigmoid(pre_activation_values)
        return sigmoid_product * (1 - sigmoid_product)

    @staticmethod
    def relu(
        pre_activation_value: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        return np.maximum(0, pre_activation_value)

    @staticmethod
    def relu_derivative(
        pre_activation_value: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        return np.where(pre_activation_value > 0, 1.0, 0.0)
