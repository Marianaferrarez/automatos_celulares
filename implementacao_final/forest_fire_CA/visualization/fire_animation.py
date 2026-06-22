import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

STATE_COLORS = ["#7A7A7A", "#228B22", "#FF6600", "#3B2A1A"]
STATE_LABELS = ["NONFLAMMABLE", "TREE", "BURNING", "BURNED"]


class FireAnimation:
    @staticmethod
    def animate(frames):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        cmap = mcolors.ListedColormap(STATE_COLORS)
        norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

        fig, ax = plt.subplots()

        img = ax.imshow(
            frames[0],
            cmap=cmap,
            norm=norm,
            animated=True,
        )

        patches = [
            mpatches.Patch(color=c, label=l)
            for c, l in zip(STATE_COLORS, STATE_LABELS)
        ]
        ax.legend(
            handles=patches,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=4,
            fontsize=8,
            frameon=False,
        )
        ax.axis("off")

        def update(frame):
            img.set_array(frame)
            return [img]

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=200,
            blit=True,
        )

        ani.save(RESULTS_DIR / "fire_spread.gif")

        if plt.get_backend().lower() != "agg":
            plt.show()

        plt.close()
