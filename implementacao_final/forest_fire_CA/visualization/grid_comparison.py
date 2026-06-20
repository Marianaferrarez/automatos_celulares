"""
Plota lado a lado: grid simulada pelo CA-Transformer vs grid de referência.
Também exibe o mapa de diferenças (overlay qualitativo — Fig. 6/8 do artigo).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Cores: NONFLAMMABLE=cinza, TREE=verde, BURNING=laranja, BURNED=marrom
STATE_COLORS = {
    0: "#7A7A7A",   # NONFLAMMABLE — cinza (rocha/água/clareira)
    1: "#228B22",   # TREE — verde floresta
    2: "#FF6600",   # BURNING — laranja
    3: "#3B2A1A",   # BURNED — marrom escuro
}


def _to_rgb(grid):
    """Converte grid de estados em imagem RGB."""
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for state, hex_color in STATE_COLORS.items():
        mask = grid == state
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        img[mask] = [r, g, b]
    return img


def _diff_map(simulated, reference):
    """
    Mapa de diferenças:
      1  = acerto (ambos iguais)
      0  = erro
    Equivale à 'difference layer' da Fig. 6 do artigo.
    """
    return (simulated == reference).astype(int)


class GridComparison:

    @staticmethod
    def plot(
        simulated: np.ndarray,
        reference: np.ndarray,
        metrics: dict | None = None,
        title: str = "Comparação CA",
    ) -> None:

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        diff = _diff_map(simulated, reference)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=13)

        # --- Grid de referência ---
        axes[0].imshow(_to_rgb(reference))
        axes[0].set_title("Referência")
        axes[0].axis("off")

        # --- Grid simulada ---
        axes[1].imshow(_to_rgb(simulated))
        axes[1].set_title("Simulado (CA-Transformer)")
        axes[1].axis("off")

        # --- Mapa de diferenças ---
        axes[2].imshow(
            diff,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
        )
        axes[2].set_title("Diferenças (verde=acerto)")
        axes[2].axis("off")

        # Legenda de estados
        patches = [
            mpatches.Patch(color=STATE_COLORS[0], label="NONFLAMMABLE"),
            mpatches.Patch(color=STATE_COLORS[1], label="TREE"),
            mpatches.Patch(color=STATE_COLORS[2], label="BURNING"),
            mpatches.Patch(color=STATE_COLORS[3], label="BURNED"),
        ]
        axes[0].legend(
            handles=patches,
            loc="lower left",
            fontsize=8,
            framealpha=0.7,
        )

        if metrics:
            info = (
                f"Accuracy={metrics['accuracy']:.3f}  "
                f"Kappa={metrics['kappa']:.3f}  "
                f"IoU={metrics['iou']:.3f}"
            )
            fig.text(
                0.5, 0.01, info,
                ha="center", fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "grid_comparison.png", dpi=150)

        if plt.get_backend().lower() != "agg":
            plt.show()

        plt.close()
