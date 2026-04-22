from dataclasses import dataclass

import numpy as np


@dataclass
class Data:
    y_true_label: float
    x_input_vector: np.ndarray
