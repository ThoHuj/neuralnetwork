from typing import Any

from torch import Tensor, randn_like
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore

from classes.training_configuration import Configuration


class AddGaussianNoise:
    def __init__(self, mean: float = 0.00, std: float = 0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: Tensor) -> Tensor:
        return tensor + randn_like(tensor) * self.std + self.mean


class DataGenerator:
    def __init__(self, config: Configuration):
        self.batch_size = config.batch_size

        training_steps: list[Any] = []
        if config.augmentation_use_affine:
            training_steps.append(
                transforms.RandomAffine(
                    degrees=config.augmentation_affine_degrees,
                    translate=config.augmentation_affine_translate,
                    scale=config.augmentation_affine_scale,
                    shear=config.augmentation_affine_shear,
                )
            )
        if config.augmentation_use_colorjitter:
            training_steps.append(
                transforms.ColorJitter(
                    brightness=config.augmentation_colorjitter_brightness,
                    contrast=config.augmentation_colorjitter_contrast,
                    saturation=config.augmentation_colorjitter_saturation,
                    hue=config.augmentation_colorjitter_hue,
                )
            )
        if config.augmentation_use_randomhorizontalflip:
            training_steps.append(transforms.RandomHorizontalFlip())
        if config.augmentation_use_gaussianblur:
            training_steps.append(
                transforms.GaussianBlur(
                    kernel_size=config.augmentation_gaussianblur_kernelsize,
                    sigma=config.augmentation_gaussianblur_sigma,
                )
            )

        training_steps.append(transforms.ToTensor())
        training_steps.append(transforms.Normalize((0.1307,), (0.3081,)))
        if config.augmentation_use_gaussiannoise:
            training_steps.append(
                AddGaussianNoise(
                    mean=config.augmentation_gaussiannoise_mean,
                    std=config.augmentation_gaussiannoise_standarddeviation,
                )
            )
        if config.augmentation_use_randomerasing:
            training_steps.append(
                transforms.RandomErasing(
                    p=config.augmentation_randomerasing_probability,
                    scale=config.augmentation_randomerasing_scale,
                )
            )

        train_transform = transforms.Compose(training_steps)
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=test_transform
        )
        self.train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
