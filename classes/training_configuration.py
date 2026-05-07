from dataclasses import dataclass


@dataclass
class TrainingConfiguration:
    run_name: str
    epochs: int
    learning_rate: float
    weight_decay: float
