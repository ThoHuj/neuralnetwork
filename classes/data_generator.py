from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore


class DataGenerator:
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        self.train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
