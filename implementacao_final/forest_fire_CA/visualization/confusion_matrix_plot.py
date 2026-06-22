import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    ConfusionMatrixDisplay
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

DISPLAY_LABELS = [
    "TREE",
    "BURNING",
    "BURNED"
]


class ConfusionMatrixPlot:
    @staticmethod
    def plot(cm):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=DISPLAY_LABELS
        )

        disp.plot()

        plt.title(
            "Matriz de Confusão"
        )

        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / "confusion_matrix.png"
        )

        if plt.get_backend().lower() != "agg":
            plt.show()

        plt.close()
