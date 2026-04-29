from torch import Tensor, tensor


class InputManager:
    def prompt_for_x_input_data(self) -> Tensor:
        while True:
            try:
                data_values = [
                    float(input(f"Enter vector value {i + 1}: ")) for i in range(2)
                ]
                if sum(data_values) > 9999999:
                    continue
            except Exception:
                continue  # Keep asking if a typo is made
            return tensor(data_values)

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
