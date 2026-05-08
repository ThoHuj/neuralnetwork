import logging
import time
from dataclasses import asdict
from typing import Literal, cast

import mlflow
import mlflow.pytorch
import torchvision  # type: ignore
from torch import Tensor, __version__, cuda, nn, no_grad, optim, zeros
from torch.utils.data import DataLoader

from classes.training_configuration import Configuration

logging.getLogger("mlflow.pytorch").setLevel(logging.ERROR)


def activation_function(name: Literal["relu"]) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


class Model(nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()
        self.device = "cuda" if cuda.is_available() else "cpu"

        block_count = len(config.convolutional_out_channels)
        expected_lengths = (
            len(config.convolutional_in_channels),
            len(config.convolutional_kernel_sizes),
            len(config.convolutional_paddings),
        )
        if len(set((block_count,) + expected_lengths)) != 1:
            raise ValueError(
                "convolutional_in_channels, convolutional_out_channels, "
                "convolutional_kernel_sizes, and convolutional_paddings "
                "must all have the same length."
            )

        feature_layers: list[nn.Module] = []
        for block_index in range(block_count):
            feature_layers.append(
                nn.Conv2d(
                    in_channels=config.convolutional_in_channels[block_index],
                    out_channels=config.convolutional_out_channels[block_index],
                    kernel_size=config.convolutional_kernel_sizes[block_index],
                    padding=config.convolutional_paddings[block_index],
                )
            )
            if config.batch_normalization:
                feature_layers.append(
                    nn.BatchNorm2d(config.convolutional_out_channels[block_index])
                )
            feature_layers.append(activation_function(config.activation_function))
            feature_layers.append(nn.MaxPool2d(2))

        self.features = nn.Sequential(*feature_layers)

        with no_grad():
            dummy_input = zeros(
                1,
                config.convolutional_in_channels[0],
                config.image_size,
                config.image_size,
            )
            flattened_size = int(self.features(dummy_input).view(1, -1).size(1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, config.classifier_hidden_neurons),
            activation_function(config.activation_function),
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(config.classifier_hidden_neurons, config.num_classes),
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
        configuration: Configuration,
    ) -> None:
        mlflow.set_experiment(  # pyright: ignore[reportUnknownMemberType]
            configuration.mlflow_experiment_name
        )

        with mlflow.start_run(run_name=configuration.run_name):
            mlflow.log_params(asdict(configuration))

            self.train()
            loss_function = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                self.parameters(),
                lr=configuration.learning_rate,
                weight_decay=configuration.weight_decay,
            )

            best_validation_accuracy = 0.0

            for epoch in range(configuration.epochs):
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

                validation_loss, validation_accuracy = self.evaluate(
                    test_loader, loss_function
                )

                mlflow.log_metrics(
                    {
                        "training_loss": average_loss,
                        "validation_loss": validation_loss,
                        "validation_accuracy": validation_accuracy,
                        "epoch_duration": epoch_duration,
                    },
                    step=epoch,
                )

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
                    f"Epoch {epoch + 1}/{configuration.epochs}",
                    f"Loss: {average_loss:.4f}",
                    f"Validation accuracy: {validation_accuracy:.2%}",
                    f"Epoch duration: {epoch_duration:.1f} seconds",
                )

            print()

            mlflow.log_metric("best_validation_accuracy", best_validation_accuracy)
