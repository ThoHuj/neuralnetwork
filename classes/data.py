from dataclasses import dataclass

import numpy as np


@dataclass
class Data:
    y_true_label: np.ndarray
    x_input_vector: np.ndarray
