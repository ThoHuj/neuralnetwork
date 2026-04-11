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
