from torch import Tensor

from classes.data_generator import DataGenerator
from classes.data_plotter import DataPlotter
from classes.input_manager import InputManager
from classes.model import Model


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
                    x_input_data: Tensor = self.input_manager.prompt_for_x_input_data()
                    a_activation_array: Tensor = self.model(x_input_data)
                    a_activation_array = a_activation_array.detach().cpu()
                    print("\rPredicions: ", end="")
                    for index, probability in enumerate(a_activation_array.flatten()):
                        predicted_class = "Bright" if probability < 0.5 else "Dark"
                        print(
                            "\rOutput number: ",
                            index,
                            "\nPrediction: ",
                            predicted_class,
                            "\nProbability:",
                            probability,
                            end="",
                        )

                case "2":
                    epochs = self.input_manager.prompt_for_integer(
                        prompt="\rEnter a number of data sets to train with: "
                    )

                    loss_history = self.model.train_model(epochs, self.data_generator)
                    self.data_plotter.plot_loss_history(loss_history)
                case "q":
                    self.exit = True
                case _:
                    print("\rBad input", end="")
