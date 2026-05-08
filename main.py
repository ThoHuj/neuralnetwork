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


def main():
    """Constructs instances and run user interface."""
    model = Model(DEFAULT_CONFIGURATION)
    print(next(model.parameters()).device)

    input_manager = InputManager()
    data_generator = DataGenerator(config=DEFAULT_CONFIGURATION)
    user_interface = UserInterface(
        input_manager,
        model,
        data_generator,
        train_conf=DEFAULT_CONFIGURATION,
    )

    user_interface.run()


if __name__ == "__main__":
    main()
