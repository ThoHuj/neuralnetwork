from torch import Tensor, randn_like
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore


class AddGaussianNoise:
    def __init__(self, mean: float = 0.00, std: float = 0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: Tensor) -> Tensor:
        return tensor + randn_like(tensor) * self.std + self.mean


class DataGenerator:
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        train_transform = transforms.Compose(
            [
                # transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
                ),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # transforms.RandomHorizontalFlip,
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                AddGaussianNoise(mean=0.0, std=0.05),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
            ]
        )
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
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
