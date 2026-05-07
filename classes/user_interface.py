from torch import nn

from classes.data_generator import DataGenerator
from classes.data_plotter import DataPlotter
from classes.input_manager import InputManager
from classes.model import Model
from classes.training_configuration import TrainingConfiguration


class UserInterface:
    exit = False
    input_manager: InputManager
    model: Model
    data_plotter: DataPlotter
    data_generator: DataGenerator

    def __init__(
        self,
        input_manager: InputManager,
        model: Model,
        data_plotter: DataPlotter,
        data_generator: DataGenerator,
    ):
        self.input_manager = input_manager
        self.model = model
        self.data_plotter = data_plotter
        self.data_generator = data_generator

    def run(self):
        while self.exit is False:
            menu_choice = self.input_manager.prompt_for_string(
                "1 - Predict\n2 - Train\nq - quit\nChoice: "
            )
            match menu_choice:
                case "1":
                    average_loss, accuracy = self.model.evaluate(
                        self.data_generator.test_loader,
                        loss_function=nn.CrossEntropyLoss(),
                    )
                    print(
                        f"Model accuracy: {accuracy:.2%}",
                        f"Average loss: {average_loss:.2f}",
                    )

                case "2":
                    epochs = self.input_manager.prompt_for_integer(
                        prompt="Enter a number of data sets to train with: "
                    )
                    training_config = TrainingConfiguration(
                        run_name="default",
                        epochs=epochs,
                        learning_rate=0.001,
                        weight_decay=1e-4,
                    )
                    self.model.train_model(
                        self.data_generator.train_loader,
                        self.data_generator.test_loader,
                        training_config,
                    )
                case "q":
                    self.exit = True
                case _:
                    print("Bad input")
