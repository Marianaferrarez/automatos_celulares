import pandas as pd
import numpy as np
from pathlib import Path

from simulation.automata import ForestCA, TREE, BURNING, BURNED, NONFLAMMABLE
from data.dataset_generator import DatasetGenerator
from data.neighborhood import (
    extract_neighborhood_sequence,
    FEATURE_NAMES,
    SEQ_LEN,
)
from visualization.fire_animation import FireAnimation

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

def _flatten_sequence(sequence):
    flat = {}
    for pos in range(SEQ_LEN):
        for f_idx, fname in enumerate(FEATURE_NAMES):
            flat[f"pos{pos}_{fname}"] = sequence[pos, f_idx]
    return flat

class DatasetBuilder:
    def __init__(
        self,
        rows=100,
        cols=100,
        steps=30,
        n_simulations=8,
        generation_threshold=0.15,
    ):
        self.rows = rows
        self.cols = cols
        self.steps = steps
        self.n_simulations = n_simulations
        self.generation_threshold = generation_threshold

    def _build_single_simulation(self, environment):
        ca = ForestCA(
            environment=environment,
            rows=self.rows,
            cols=self.cols,
            model=None,
            scaler=None,
            threshold=self.generation_threshold,
        )

        samples = []

        for _ in range(self.steps):

            current_grid = ca.grid.copy()
            ca.step()
            next_grid = ca.grid.copy()

            for i in range(self.rows):
                for j in range(self.cols):

                    current_state = current_grid[i, j]

                    if current_state in (BURNED, NONFLAMMABLE):
                        continue

                    has_burning_neighbor = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if (
                                0 <= ni < self.rows
                                and 0 <= nj < self.cols
                                and current_grid[ni, nj] == BURNING
                            ):
                                has_burning_neighbor = True
                                break
                        if has_burning_neighbor:
                            break

                    if current_state == TREE and not has_burning_neighbor:
                        continue

                    sequence = extract_neighborhood_sequence(
                        environment, current_grid, i, j
                    )
                    row = _flatten_sequence(sequence)
                    row["next_state"] = next_grid[i, j]

                    samples.append(row)

        return samples

    def build(self):
        all_samples = []

        for sim_idx in range(self.n_simulations):

            generator = DatasetGenerator(self.rows, self.cols)
            environment = generator.generate_all()

            samples = self._build_single_simulation(environment)
            all_samples.extend(samples)

        df = pd.DataFrame(all_samples)

        return df

    def preview_animation(self, steps=50):
        generator = DatasetGenerator(self.rows, self.cols)
        environment = generator.generate_all()

        ca = ForestCA(
            environment=environment,
            rows=self.rows,
            cols=self.cols,
            model=None,
            scaler=None,
            threshold=self.generation_threshold,
        )

        frames = []
        for _ in range(steps):
            frames.append(ca.grid.copy())
            ca.step()

        FireAnimation.animate(frames)
