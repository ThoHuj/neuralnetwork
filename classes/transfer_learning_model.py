import copy
import logging
import time
from dataclasses import asdict
from typing import cast

import mlflow
import mlflow.pytorch
import torchvision  # type: ignore
from torch import Tensor, __version__, cuda, nn, no_grad, optim
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50  # type: ignore

from classes.training_configuration import Configuration

logging.getLogger("mlflow.pytorch").setLevel(logging.ERROR)

RESNET50_BACKBONE_OUTPUT_FEATURES = 2048


class TransferLearningModel(nn.Module):
    def __init__(self, configuration: Configuration) -> None:
        super().__init__()
        self.device = "cuda" if cuda.is_available() else "cpu"

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        for parameter in backbone.parameters():
            parameter.requires_grad = False

        backbone.fc = nn.Identity()  # type: ignore[assignment]
        self.backbone = backbone

        self.classifier_head = nn.Sequential(
            nn.Dropout(p=configuration.dropout_rate),
            nn.Linear(RESNET50_BACKBONE_OUTPUT_FEATURES, configuration.num_classes),
        )

        self.to(self.device)

        trainable_count = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        total_count = sum(parameter.numel() for parameter in self.parameters())
        print(
            f"TransferLearningModel: {trainable_count:,} trainable"
            f" / {total_count:,} total parameters"
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.device)
        features: Tensor = self.backbone(x)
        output: Tensor = self.classifier_head(features)
        return output

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
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=configuration.learning_rate,
                weight_decay=configuration.weight_decay,
            )

            best_validation_accuracy = 0.0
            best_state_dict: dict[str, Tensor] | None = None

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
                    best_state_dict = copy.deepcopy(self.state_dict())

                print(
                    f"Epoch {epoch + 1}/{configuration.epochs}",
                    f"Loss: {average_loss:.4f}",
                    f"Validation accuracy: {validation_accuracy:.2%}",
                    f"Epoch duration: {epoch_duration:.1f} seconds",
                )

            print()

            if best_state_dict is not None:
                self.load_state_dict(best_state_dict)
                mlflow.pytorch.log_model(  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage]
                    self,
                    serialization_format="pickle",
                    pip_requirements=[
                        f"torch=={__version__}",
                        f"torchvision=={torchvision.__version__}",
                    ],
                    name="best_accuracy",
                )

            mlflow.log_metric("best_validation_accuracy", best_validation_accuracy)
