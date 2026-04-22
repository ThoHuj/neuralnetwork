from dataclasses import dataclass

from numpy import ndarray


@dataclass
class Data:
    label: ndarray
    vector: ndarray
