import matplotlib.pyplot as plt

import matplotlib.animation as animation
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


class FireAnimation:
    
    @staticmethod
    def animate(frames):

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots()

        img = ax.imshow(
            frames[0],
            cmap="hot",
            animated=True
        )

        def update(frame):

            img.set_array(frame)

            return [img]

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=200,
            blit=True
        )

        ani.save(
            RESULTS_DIR / "fire_spread.gif"
        )

        if plt.get_backend().lower() != "agg":
            plt.show()

        plt.close()