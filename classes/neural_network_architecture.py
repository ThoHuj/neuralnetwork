class NeuralNetworkArchitecture:
    def __init__(self):
        self.layer_structure: list[dict[str, int | str]] = [
            {
                "input_dimensions": 2,
                "output_dimensions": 400,
                "activation_function": "relu",
            },
            {
                "input_dimensions": 400,
                "output_dimensions": 400,
                "activation_function": "relu",
            },
            {
                "input_dimensions": 400,
                "output_dimensions": 400,
                "activation_function": "relu",
            },
            {
                "input_dimensions": 400,
                "output_dimensions": 1,
                "activation_function": "sigmoid",
            },
        ]
