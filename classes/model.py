from typing import cast

from torch import Tensor, cuda, nn, optim
from torch.utils.data import DataLoader


class Model(nn.Module):
    DEFAULT_LEARNING_RATE = 0.001

    def __init__(self):
        super().__init__()
        self.device = "cuda" if cuda.is_available() else "cpu"

        self.features = nn.Sequential(
            # input shape is 1x28x28 (1 grayscale channel, 28x28 pixels per image)
            # Convolutional block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Convolutional block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 10),
        )
        self.to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.device)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def train_model(
        self, epochs: int, train_loader: DataLoader[tuple[Tensor, Tensor]]
    ) -> list[float]:
        self.train()
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
                loss = cast(Tensor, loss_function(output, labels))
                loss.backward()  # pyright: ignore[reportUnknownMemberType]
                optimizer.step()  # pyright: ignore[reportUnknownMemberType]

                running_loss += loss.item()

            average_loss = running_loss / len(train_loader)
            loss_history.append(average_loss)
            print(f"\rEpoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}", end="")

        print()
        return loss_history
