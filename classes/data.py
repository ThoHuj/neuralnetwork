class Data:
    name: str
    vector: list[float]
    truth: dict[str, float]

    def __init__(
        self, name: str, data_vector: list[float], truth: dict[str, float]
    ) -> None:
        self.name = name
        self.vector = data_vector
