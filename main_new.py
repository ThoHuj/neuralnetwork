from classes.data_generator import DataGenerator
from classes.data_plotter import DataPlotter
from classes.input_manager import InputManager
from classes.model import Model
from classes.user_interface import UserInterface


def main():
    """Constructs instances and run user interface."""
    model = Model()
    print(next(model.parameters()).device)

    input_manager = InputManager()
    data_plotter = DataPlotter()
    data_generator = DataGenerator()
    user_interface = UserInterface(input_manager, model, data_plotter, data_generator)

    user_interface.run()


if __name__ == "__main__":
    main()
