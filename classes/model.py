import logging
import time
from dataclasses import asdict
from typing import cast

import mlflow
import mlflow.pytorch
import torchvision  # type: ignore
from torch import Tensor, __version__, cuda, nn, no_grad, optim
from torch.utils.data import DataLoader

from classes.training_configuration import TrainingConfiguration

# This prevents mlflow warnings from appearing in the terminal, but allows errors to be printed
logging.getLogger("mlflow.pytorch").setLevel(logging.ERROR)


class Model(nn.Module):
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

    def evaluate(
        self,
        data_loader: DataLoader[tuple[Tensor, Tensor]],
        loss_function: nn.Module,
    ) -> tuple[float, float]:
        self.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with no_grad():
            for images, labels in data_loader:
                images: Tensor
                labels: Tensor
                images, labels = images.to(self.device), labels.to(self.device)
                output: Tensor = self(images)
                total_loss += loss_function(output, labels).item()
                predictions = output.argmax(dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)

        average_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_samples
        self.train()
        return average_loss, accuracy

    def train_model(
        self,
        train_loader: DataLoader[tuple[Tensor, Tensor]],
        test_loader: DataLoader[tuple[Tensor, Tensor]],
        training_config: TrainingConfiguration,
    ) -> None:
        # Set project
        mlflow.set_experiment("mnist-cnn")  # pyright: ignore[reportUnknownMemberType]

        with mlflow.start_run(run_name=training_config.run_name):
            # Log hyperparameters
            mlflow.log_params(asdict(training_config))

            self.train()
            loss_function = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                self.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
            )

            best_validation_accuracy = 0.0

            for epoch in range(training_config.epochs):
                # Start logging epoch duration
                epoch_start = time.perf_counter()
                running_loss = 0.0

                self.train()
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
                epoch_duration = time.perf_counter() - epoch_start

                # Validation of epoch result
                validation_loss, validation_accuracy = self.evaluate(
                    test_loader, loss_function
                )

                # Log epoch results
                mlflow.log_metrics(
                    {
                        "training_loss": average_loss,
                        "validation_loss": validation_loss,
                        "validation_accuracy": validation_accuracy,
                        "epoch_duration": epoch_duration,
                    },
                    step=epoch,
                )

                # Save checkpoint when model improves
                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    mlflow.pytorch.log_model(  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage]
                        self,
                        serialization_format="pickle",
                        pip_requirements=[
                            f"torch=={__version__}",
                            f"torchvision=={torchvision.__version__}",
                        ],
                        name="best_accuracy",
                    )

                print(
                    f"\rEpoch {epoch + 1}/{training_config.epochs}",
                    f"Loss: {average_loss:.4f}",
                    f"Validation accuracy: {validation_accuracy:.2%}",
                    f"Epoch duration: {epoch_duration:.1f} seconds",
                )

            print()

            mlflow.log_metric("best_validation_accuracy", best_validation_accuracy)
