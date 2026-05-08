from classes.data_generator import DataGenerator
from classes.input_manager import InputManager
from classes.model import Model
from classes.training_configuration import Configuration
from classes.user_interface import UserInterface

DEFAULT_CONFIGURATION = Configuration(
    run_name="run5",
    epochs=1,
    learning_rate=0.1,
    weight_decay=1e-2,
    dropout_rate=0.5,
    batch_size=300,
    classifier_hidden_neurons=40,
    convolutional_in_channels=[1, 32],
    convolutional_out_channels=[32, 14],
    convolutional_kernel_sizes=[3, 3],
    convolutional_paddings=[1, 1],
    activation_function="relu",
    batch_normalization=True,
    dataset_kind="mnist",
    image_size=28,
    num_classes=10,
    normalize_mean=(0.1307,),
    normalize_std=(0.3081,),
    data_root_mnist="./data",
    image_folder_train_dir="",
    image_folder_val_dir="",
    download_cats_dogs_filtered_if_missing=False,
    mlflow_experiment_name="mnist-cnn",
    # Augmentation configuration
    augmentation_use_affine=False,
    augmentation_affine_degrees=10,
    augmentation_affine_translate=(0.1, 0.1),
    augmentation_affine_scale=(0.9, 1.1),
    augmentation_affine_shear=10,
    augmentation_use_colorjitter=False,
    augmentation_colorjitter_brightness=0.2,
    augmentation_colorjitter_contrast=0.2,
    augmentation_colorjitter_saturation=0.2,
    augmentation_colorjitter_hue=0.1,
    augmentation_use_gaussianblur=False,
    augmentation_gaussianblur_kernelsize=3,
    augmentation_gaussianblur_sigma=(0.1, 0.5),
    augmentation_use_gaussiannoise=False,
    augmentation_gaussiannoise_mean=0.0,
    augmentation_gaussiannoise_standarddeviation=0.05,
    augmentation_use_randomerasing=True,
    augmentation_randomerasing_probability=0.3,
    augmentation_randomerasing_scale=(0.02, 0.15),
    augmentation_use_randomhorizontalflip=False,
)

CATS_DOGS_CONFIGURATION = Configuration(
    run_name="cats_dogs_run13",
    epochs=1,
    # AdamW on RGB needs much smaller lr than MNIST-style 0.1; otherwise logits stay
    # tiny, softmax stays ~uniform, loss stays ~ln(2) and val accuracy stays ~50%.
    learning_rate=0.0001,
    weight_decay=1e-4,
    dropout_rate=0.25,
    batch_size=32,
    classifier_hidden_neurons=800,
    convolutional_in_channels=[3, 32, 160],
    convolutional_out_channels=[32, 160, 32],
    convolutional_kernel_sizes=[3, 3, 3],
    convolutional_paddings=[1, 1, 1],
    activation_function="relu",
    batch_normalization=False,
    dataset_kind="image_folder",
    image_size=128,
    num_classes=2,
    normalize_mean=(0.485, 0.456, 0.406),
    normalize_std=(0.229, 0.224, 0.225),
    data_root_mnist="./data",
    image_folder_train_dir="./data/cats_dogs/train",
    image_folder_val_dir="./data/cats_dogs/val",
    download_cats_dogs_filtered_if_missing=True,
    mlflow_experiment_name="cats-dogs-cnn",
    augmentation_use_affine=False,
    augmentation_affine_degrees=10,
    augmentation_affine_translate=(0.1, 0.1),
    augmentation_affine_scale=(0.9, 1.1),
    augmentation_affine_shear=10,
    augmentation_use_colorjitter=False,
    augmentation_colorjitter_brightness=0.2,
    augmentation_colorjitter_contrast=0.2,
    augmentation_colorjitter_saturation=0.2,
    augmentation_colorjitter_hue=0.1,
    augmentation_use_gaussianblur=False,
    augmentation_gaussianblur_kernelsize=3,
    augmentation_gaussianblur_sigma=(0.1, 0.5),
    augmentation_use_gaussiannoise=False,
    augmentation_gaussiannoise_mean=0.0,
    augmentation_gaussiannoise_standarddeviation=0.05,
    augmentation_use_randomerasing=False,
    augmentation_randomerasing_probability=0.3,
    augmentation_randomerasing_scale=(0.02, 0.15),
    augmentation_use_randomhorizontalflip=True,
)

ACTIVE_CONFIGURATION = CATS_DOGS_CONFIGURATION
# ACTIVE_CONFIGURATION = DEFAULT_CONFIGURATION


def main():
    """Constructs instances and run user interface."""
    model = Model(ACTIVE_CONFIGURATION)
    print(next(model.parameters()).device)

    input_manager = InputManager()
    data_generator = DataGenerator(config=ACTIVE_CONFIGURATION)
    user_interface = UserInterface(
        input_manager,
        model,
        data_generator,
        train_conf=ACTIVE_CONFIGURATION,
    )

    user_interface.run()


if __name__ == "__main__":
    main()
