from dataclasses import replace

from torch import nn

from classes.data_generator import DataGenerator
from classes.input_manager import InputManager
from classes.model import Model
from classes.training_configuration import Configuration


class UserInterface:
    exit = False
    input_manager: InputManager
    model: Model
    data_generator: DataGenerator
    train_conf: Configuration

    def __init__(
        self,
        input_manager: InputManager,
        model: Model,
        data_generator: DataGenerator,
        train_conf: Configuration,
    ):
        self.input_manager = input_manager
        self.model = model
        self.data_generator = data_generator
        self.train_conf = train_conf

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
                        prompt="Enter a number of epochs: "
                    )
                    run_configuration = replace(
                        self.train_conf,
                        epochs=epochs,
                    )
                    self.model.train_model(
                        self.data_generator.train_loader,
                        self.data_generator.test_loader,
                        run_configuration,
                    )
                case "q":
                    self.exit = True
                case _:
                    print("Bad input")
