from torch import Tensor, cuda, nn, optim
from torch.utils.data import DataLoader


class Model(nn.Module):
    DEFAULT_LEARNING_RATE = 0.001

    def __init__(self):
        super().__init__()
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
        )
        self.to(self.device)

    def forward(self, x_input_data: Tensor) -> Tensor:
        x_input_data = x_input_data.to(self.device)
        return self.linear_stack(x_input_data)

    def train_model(
        self, epochs: int, train_loader: DataLoader[tuple[Tensor, Tensor]]
    ) -> list[float]:
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.DEFAULT_LEARNING_RATE)
        loss_history: list[float] = []

        for epoch in range(epochs):
            running_loss = 0.0

            for images, labels in train_loader:
                images: Tensor
                labels: Tensor
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output: Tensor = self(images)
                loss: Tensor = loss_function(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            average_loss = running_loss / len(train_loader)
            loss_history.append(average_loss)
            print(f"\rEpoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}", end="")

        print()
        return loss_history
