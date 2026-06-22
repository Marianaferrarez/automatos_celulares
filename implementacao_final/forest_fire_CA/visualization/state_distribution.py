import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

STATE_LABELS = {
    1: "TREE",
    2: "BURNING",
    3: "BURNED"
}

class StateDistribution:
    @staticmethod
    def plot(df):

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        counts = df["next_state"].value_counts().reindex([1, 2, 3], fill_value=0)
        counts.index = [STATE_LABELS[state] for state in counts.index]

        plt.figure(figsize=(8, 5))

        counts.plot(kind="bar")

        plt.title("Distribuição dos Estados Futuros")

        plt.xlabel("Estado")

        plt.ylabel("Quantidade")

        plt.grid(axis="y")

        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / "state_distribution.png"
        )

        if plt.get_backend().lower() != "agg":
            plt.show()

        plt.close()