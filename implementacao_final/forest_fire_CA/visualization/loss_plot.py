import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

class LossPlot:
    
    @staticmethod
    def plot(losses):

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 5))

        plt.plot(losses)

        plt.title(
            "Loss por Época"
        )

        plt.xlabel(
            "Época"
        )

        plt.ylabel(
            "Loss"
        )

        plt.grid()

        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / "loss_curve.png"
        )

        if plt.get_backend().lower() != "agg":
            plt.show()

        plt.close()