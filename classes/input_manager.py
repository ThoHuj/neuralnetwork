import numpy as np

from classes.data import Data


class InputManager:
    def prompt_for_data(self, enter_label: bool = True) -> Data:
        while True:
            try:
                if enter_label:
                    user_input = input(
                        "Enter this vector's label (the correct answer): "
                    )
                    label = float(user_input)
                else:
                    label = 0.0
                data_vector = [
                    float(input(f"Enter vector value {i}: ")) for i in range(2)
                ]
                if sum(data_vector) > 9999999:
                    continue
            except Exception:
                continue  # Keep asking if a typo is made
            data_vector = np.array(data_vector)
            return Data(label, data_vector)

    def prompt_for_integer(self, prompt: str) -> int:
        while True:
            try:
                user_input = int(input(prompt))
            except ValueError:
                print("please enter a correct value.")
                continue
            return user_input

    def prompt_for_yes_no(self, prompt: str) -> bool:
        """Returns True if answer is 'yes'."""
        while True:
            user_input = input(prompt).lower()
            if user_input not in ["y", "n"]:
                print("please enter a correct value.")
                continue
            return True if user_input == "y" else False

    def prompt_for_string(self, prompt: str) -> str:
        user_input = input(prompt)
        return user_input
