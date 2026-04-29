from torch import Tensor, cuda, nn, optim

from classes.data_generator import DataGenerator


class Model(nn.Module):
    DEFAULT_LEARNING_RATE = 0.1

    def __init__(self):
        super().__init__()
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.linear_stack = nn.Sequential(
            nn.Linear(2, 20),
            nn.Sigmoid(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )
        self.to(self.device)

    def forward(self, x_input_data: Tensor) -> Tensor:
        x_input_data = x_input_data.to(self.device)
        a_activation: Tensor = self.linear_stack(x_input_data)
        return a_activation

    def train_model(self, epochs: int, data_generator: DataGenerator) -> list[float]:
        loss_function = nn.BCELoss()
        loss_history: list[float] = []
        optimizer = optim.Adam(self.parameters(), lr=self.DEFAULT_LEARNING_RATE)
        for epoch in range(epochs):
            print(f"\r{epoch} of {epochs}", end="")
            x_input_vector, y_true_label_vector = (
                data_generator.generate_random_image_data(
                    batch_size=100000, device=self.device
                )
            )

            optimizer.zero_grad()
            a_activation_vector = self(x_input_vector)
            loss: Tensor = loss_function(a_activation_vector, y_true_label_vector)
            loss.backward()  # type: ignore
            optimizer.step()  # type: ignore
            loss_history.append(loss.item())
        return loss_history
