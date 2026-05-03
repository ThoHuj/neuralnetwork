import matplotlib

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import matplotx  # type: ignore

plt.style.use(matplotx.styles.pitaya_smoothie["dark"])  # type: ignore


class DataPlotter:
    def plot_loss_history(self, loss_history: list[float]) -> None:
        plt.plot(loss_history, marker="o")  # type: ignore
        plt.show()  # type: ignore
