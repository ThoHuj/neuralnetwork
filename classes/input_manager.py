from classes.data import Data


class InputManager:
    def prompt_for_data(self) -> Data:
        while True:
            try:
                user_input = input("Enter this vector's label (the correct answer): ")
                if user_input == "r":
                    continue
                label = float(user_input)
                data_vector = [
                    float(input(f"Enter vector value {i}: ")) for i in range(2)
                ]
                if sum(data_vector) > 9999999:
                    continue
            except Exception:
                continue  # Keep asking if a typo is made
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
