from torch import nn, Tensor, optim
from classes.data_generator import DataGenerator


class Model(nn.Module):
    DEFAULT_LEARNING_RATE = 0.01

    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(2, 400),
            nn.ReLU(),
            nn.Linear(400, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_input_data: Tensor) -> Tensor:
        a_activation: Tensor = self.linear_stack(x_input_data)
        return a_activation

    def train_model(self, epochs: int, data_generator: DataGenerator) -> list[float]:
        loss_function = nn.BCELoss()
        loss_history: list[float] = []
        optimizer = optim.Adam(self.parameters(), lr=self.DEFAULT_LEARNING_RATE)
        for epoch in range(epochs):
            x_input_vector, y_true_label_vector = (
                data_generator.generate_random_image_data(batch_size=100000)
            )
            optimizer.zero_grad()
            a_activation_vector = self(x_input_vector)
            loss: Tensor = loss_function(a_activation_vector, y_true_label_vector)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        return loss_history
