from classes.input_manager import InputManager
from classes.network import Network
from classes.data_plotter import DataPlotter


class UserInterface:
    exit = False
    input_manager: InputManager
    network: Network
    data_plotter: DataPlotter

    def __init__(
        self, input_manager: InputManager, network: Network, data_plotter: DataPlotter
    ):
        self.input_manager = input_manager
        self.network = network
        self.data_plotter = data_plotter
        self.loss_history: list[float] = []

    def run(self):
        while self.exit is False:
            menu_choice = self.input_manager.prompt_for_string(
                "1 - Predict\n2 - Train\nq - quit\nChoice: "
            )
            match menu_choice:
                case "1":
                    data = self.input_manager.prompt_for_data(enter_label=False)
                    a_activation_array = self.network.full_forward_propagation(
                        data.x_input_vector
                    )
                    print(a_activation_array.shape[1])
                    print("Predicions: ")
                    for index, probability in enumerate(a_activation_array.flatten()):
                        predicted_class = "Bright" if probability < 0.5 else "Dark"
                        print(
                            "Output number: ",
                            index,
                            "\nPrediction: ",
                            predicted_class,
                            "\nProbability:",
                            probability,
                        )

                case "2":
                    iterations = self.input_manager.prompt_for_integer(
                        prompt="Enter a number of data sets to train with: "
                    )

                    self.loss_history += self.network.train_model(iterations)
                    self.data_plotter.plot_loss_history(self.loss_history)
                case "q":
                    self.exit = True
                case _:
                    print("Bad input")
