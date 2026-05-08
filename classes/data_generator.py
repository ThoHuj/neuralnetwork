import os
import random
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image
from torch import Tensor, randn_like
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore

from classes.training_configuration import Configuration

# Official Microsoft mirror of the Kaggle Dogs vs Cats zip (~787 MB). The former
# TensorFlow/GCS "cats_and_dogs_filtered.zip" often returns HTTP 403 for programmatic clients.
_MS_KAGGLE_CATS_DOGS_URL = (
    "https://download.microsoft.com/download/3/E/1/"
    "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
)
_MS_ZIP_NAME = "kagglecatsanddogs_5340.zip"
_MS_ZIP_MIN_BYTES = 400 * 1024 * 1024
_DOWNLOAD_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".gif", ".webp"}


def _download_url_to_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": _DOWNLOAD_USER_AGENT}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=1200) as resp, open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh, length=1024 * 1024)
    except urllib.error.HTTPError as e:
        if e.code != 403:
            raise
        import requests

        with requests.get(url, headers=headers, timeout=1200, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)


def _imagefolder_layout_ready(root: Path) -> bool:
    if not root.is_dir():
        return False
    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue
        for f in class_dir.iterdir():
            if f.is_file() and f.suffix.lower() in _IMAGE_SUFFIXES:
                return True
    return False


def _valid_jpeg_files(class_dir: Path) -> list[Path]:
    valid: list[Path] = []
    for f in sorted(class_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() != ".jpg":
            continue
        try:
            with Image.open(f) as im:
                im.load()
            valid.append(f)
        except OSError:
            continue
    return valid


def _locate_petimages_root(search_under: Path) -> Path:
    for candidate in search_under.rglob("PetImages"):
        if (
            candidate.is_dir()
            and (candidate / "Cat").is_dir()
            and (candidate / "Dog").is_dir()
        ):
            return candidate
    raise RuntimeError(
        f"Unpacked archive under {search_under} did not contain PetImages/Cat and PetImages/Dog."
    )


def _split_petimages_to_train_val(
    petimages: Path,
    train_dir: Path,
    val_dir: Path,
    *,
    val_fraction: float,
    seed: int,
) -> None:
    rnd = random.Random(seed)
    for class_name in ("Cat", "Dog"):
        files = _valid_jpeg_files(petimages / class_name)
        if len(files) < 2:
            raise RuntimeError(
                f"Too few readable .jpg images in {petimages / class_name} ({len(files)})."
            )
        rnd.shuffle(files)
        n_val = max(1, int(round(len(files) * val_fraction)))
        n_val = min(n_val, len(files) - 1)
        val_files = files[:n_val]
        train_files = files[n_val:]
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)
        for f in train_files:
            shutil.copy2(f, train_dir / class_name / f.name)
        for f in val_files:
            shutil.copy2(f, val_dir / class_name / f.name)


def _ensure_cats_dogs_filtered_dataset(train_dir: Path, val_dir: Path) -> None:
    train_dir = train_dir.resolve()
    val_dir = val_dir.resolve()
    if train_dir.parent != val_dir.parent:
        raise ValueError(
            "download_cats_dogs_filtered_if_missing requires train and val "
            "directories to share the same parent folder."
        )
    if _imagefolder_layout_ready(train_dir) and _imagefolder_layout_ready(val_dir):
        return

    root = train_dir.parent
    zip_path = root / _MS_ZIP_NAME
    unpack_dir = root / "_kaggle_cats_dogs_unpack"
    legacy_filtered = root / "cats_and_dogs_filtered"

    for p in (train_dir, val_dir, unpack_dir, legacy_filtered):
        if p.is_dir():
            shutil.rmtree(p)
        elif p.is_file():
            p.unlink()

    if not zip_path.is_file() or zip_path.stat().st_size < _MS_ZIP_MIN_BYTES:
        if zip_path.is_file():
            zip_path.unlink()
        print(
            "Downloading Microsoft Kaggle Cats vs Dogs archive (~787 MB). "
            "This runs once; the zip is kept for later runs.",
            flush=True,
        )
        _download_url_to_file(_MS_KAGGLE_CATS_DOGS_URL, zip_path)

    try:
        unpack_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(path=str(unpack_dir))
    except zipfile.BadZipFile as e:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{zip_path} is not a valid zip archive. Delete it and retry."
        ) from e

    petimages = _locate_petimages_root(unpack_dir)
    _split_petimages_to_train_val(
        petimages, train_dir, val_dir, val_fraction=0.2, seed=42
    )
    shutil.rmtree(unpack_dir, ignore_errors=True)


class AddGaussianNoise:
    def __init__(self, mean: float = 0.00, std: float = 0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: Tensor) -> Tensor:
        return tensor + randn_like(tensor) * self.std + self.mean


class DataGenerator:
    def __init__(self, config: Configuration):
        self.batch_size = config.batch_size

        if len(config.normalize_mean) != len(config.normalize_std):
            raise ValueError("normalize_mean and normalize_std must have the same length.")
        if len(config.normalize_mean) != config.convolutional_in_channels[0]:
            raise ValueError(
                "normalize tuples length must match convolutional_in_channels[0] "
                "(input channel count)."
            )

        training_steps: list[Any] = []
        if config.dataset_kind == "image_folder":
            training_steps.append(
                transforms.Resize((config.image_size, config.image_size))
            )
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
        training_steps.append(
            transforms.Normalize(config.normalize_mean, config.normalize_std)
        )
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

        test_steps: list[Any] = []
        if config.dataset_kind == "image_folder":
            test_steps.append(
                transforms.Resize((config.image_size, config.image_size))
            )
        test_steps.append(transforms.ToTensor())
        test_steps.append(
            transforms.Normalize(config.normalize_mean, config.normalize_std)
        )
        test_transform = transforms.Compose(test_steps)

        if config.dataset_kind == "mnist":
            train_dataset = datasets.MNIST(
                root=config.data_root_mnist,
                train=True,
                download=True,
                transform=train_transform,
            )
            test_dataset = datasets.MNIST(
                root=config.data_root_mnist,
                train=False,
                download=True,
                transform=test_transform,
            )
        elif config.dataset_kind == "image_folder":
            if config.download_cats_dogs_filtered_if_missing:
                _ensure_cats_dogs_filtered_dataset(
                    Path(config.image_folder_train_dir),
                    Path(config.image_folder_val_dir),
                )
            for path, label in (
                (config.image_folder_train_dir, "train"),
                (config.image_folder_val_dir, "validation"),
            ):
                if not os.path.isdir(path):
                    raise FileNotFoundError(
                        f"Image folder {label} path does not exist or is not a directory: {path!r}. "
                        "Create it with one subdirectory per class containing images, "
                        "or use dataset_kind='mnist' in your configuration."
                    ) from None
            train_dataset = datasets.ImageFolder(
                root=config.image_folder_train_dir,
                transform=train_transform,
            )
            test_dataset = datasets.ImageFolder(
                root=config.image_folder_val_dir,
                transform=test_transform,
            )
            if len(train_dataset.classes) != config.num_classes:
                raise ValueError(
                    f"Training ImageFolder has {len(train_dataset.classes)} classes "
                    f"but num_classes is {config.num_classes}."
                )
            if len(test_dataset.classes) != config.num_classes:
                raise ValueError(
                    f"Validation ImageFolder has {len(test_dataset.classes)} classes "
                    f"but num_classes is {config.num_classes}."
                )
        else:
            raise ValueError(f"Unsupported dataset_kind: {config.dataset_kind}")

        self.train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
