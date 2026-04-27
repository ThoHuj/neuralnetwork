from classes.network import Network
from classes.input_manager import InputManager
from classes.user_interface import UserInterface
from classes.data_plotter import DataPlotter
from classes.neural_network_architecture import NeuralNetworkArchitecture


def main():
    """Constructs instances and run user interface."""
    neural_network_architecture = NeuralNetworkArchitecture()
    network = Network()
    network.initialize_layers(neural_network_architecture)
    input_manager = InputManager()
    data_plotter = DataPlotter()
    user_interface = UserInterface(input_manager, network, data_plotter)

    user_interface.run()


if __name__ == "__main__":
    main()
