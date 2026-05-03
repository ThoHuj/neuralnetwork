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
                    images: Tensor
                    labels: Tensor
                    images, labels = next(iter(self.data_generator.test_loader))
                    output: Tensor = self.model(images)
                    predictions = output.argmax(dim=1).cpu()
                    labels = labels.cpu()
                    correct = (predictions == labels).sum().item()
                    total = labels.size(0)
                    print(f"Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")
                    for index in range(min(10, total)):
                        print(
                            f"  Predicted: {predictions[index].item()}, Actual: {labels[index].item()}"
                        )

                case "2":
                    epochs = self.input_manager.prompt_for_integer(
                        prompt="Enter a number of data sets to train with: "
                    )

                    loss_history = self.model.train_model(
                        epochs, self.data_generator.train_loader
                    )
                    self.data_plotter.plot_loss_history(loss_history)
                case "q":
                    self.exit = True
                case _:
                    print("Bad input")
