from dataclasses import dataclass
from torch import Tensor


@dataclass
class Data:
    y_true_label: Tensor
    x_input_vector: Tensor
