import pandas as pd
from pathlib import Path
import subprocess
import sys

from data.dataset_builder import DatasetBuilder
from visualization.state_distribution import (
    StateDistribution
)


ROOT_DIR = Path(__file__).resolve().parent

ROWS = 40
COLS = 40
STEPS = 50
ANIMATION_FRAMES = 50


def main():

    builder = DatasetBuilder(
        rows=ROWS,
        cols=COLS,
        steps=STEPS
    )

    df = builder.build()

    df.to_csv(
        ROOT_DIR / "forest_fire_dataset.csv",
        index=False
    )

    StateDistribution.plot(df)

    builder.preview_animation(
        steps=ANIMATION_FRAMES
    )

    subprocess.run(
        [sys.executable, str(ROOT_DIR / "train.py")],
        cwd=ROOT_DIR,
        check=True
    )


if __name__ == "__main__":
    main()