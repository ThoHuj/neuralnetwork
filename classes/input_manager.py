class InputManager:
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
