from dataclasses import dataclass
from typing import Literal

DatasetKind = Literal["mnist", "image_folder"]


@dataclass
class Configuration:
    run_name: str
    epochs: int
    learning_rate: float
    weight_decay: float
    dropout_rate: float
    batch_size: int
    classifier_hidden_neurons: int
    convolutional_in_channels: list[int]
    convolutional_out_channels: list[int]
    convolutional_kernel_sizes: list[int]
    convolutional_paddings: list[int]

    activation_function: Literal["relu"]

    batch_normalization: bool

    dataset_kind: DatasetKind
    image_size: int
    num_classes: int
    normalize_mean: tuple[float, ...]
    normalize_std: tuple[float, ...]
    data_root_mnist: str
    image_folder_train_dir: str
    image_folder_val_dir: str
    # If True with image_folder, missing train/val dirs are filled from the Microsoft
    # Kaggle Cats vs Dogs archive (downloaded once, then split into train/val).
    download_cats_dogs_filtered_if_missing: bool
    mlflow_experiment_name: str

    # Augmentation configuration
    augmentation_use_affine: bool
    augmentation_affine_degrees: int
    augmentation_affine_translate: tuple[float, float]
    augmentation_affine_scale: tuple[float, float]
    augmentation_affine_shear: int

    augmentation_use_colorjitter: bool
    augmentation_colorjitter_brightness: float
    augmentation_colorjitter_contrast: float
    augmentation_colorjitter_saturation: float
    augmentation_colorjitter_hue: float

    augmentation_use_gaussianblur: bool
    augmentation_gaussianblur_kernelsize: int  # Number must be odd
    augmentation_gaussianblur_sigma: tuple[float, float]

    augmentation_use_gaussiannoise: bool
    augmentation_gaussiannoise_mean: float
    augmentation_gaussiannoise_standarddeviation: float

    augmentation_use_randomerasing: bool
    augmentation_randomerasing_probability: float
    augmentation_randomerasing_scale: tuple[float, float]

    augmentation_use_randomhorizontalflip: bool
