from torch import nn, Tensor

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(2, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1),
            nn.Sigmoid()

        )

    def forward(self, x_input_data: Tensor):
        a_activation = self.linear_stack(x_input_data)
        return a_activation 