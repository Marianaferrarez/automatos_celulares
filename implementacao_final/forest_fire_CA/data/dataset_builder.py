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
    """
    Achata (seq_len, n_features) em um dict de colunas nomeadas
    pos0_temperature, pos0_humidity, ..., pos8_current_state.
    Posição 8 é sempre a célula central (ver data/neighborhood.py).
    """
    flat = {}
    for pos in range(SEQ_LEN):
        for f_idx, fname in enumerate(FEATURE_NAMES):
            flat[f"pos{pos}_{fname}"] = sequence[pos, f_idx]
    return flat


class DatasetBuilder:
    """
    Gera o dataset de treino rodando várias simulações de incêndio
    independentes, cada uma com seu próprio ambiente (elevação, vento,
    etc). Isso é essencial: se o dataset vier de uma única simulação,
    variáveis "globais" como wind_direction ficam constantes e o
    StandardScaler aprende uma média/desvio que não generaliza para
    nenhum outro ambiente (variância zero ⇒ normalização explode).

    Cada amostra agora inclui a vizinhança de Moore completa (8
    vizinhos + célula central), para que o Transformer tenha uma
    sequência real sobre a qual aplicar self-attention, em vez de uma
    única célula (o que reduzia o modelo a uma MLP disfarçada).
    """

    def __init__(
        self,
        rows=100,
        cols=100,
        steps=30,
        n_simulations=8,
        # Threshold baixo garante propagação suficiente durante a
        # geração do dataset (não é o threshold da simulação final).
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

                    # Não coleta amostras de células já queimadas nem
                    # de células sem material combustível (Regras 2 e
                    # 3 — a transição é trivial e não precisa ser
                    # aprendida pelo Transformer).
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
        """
        Roda `n_simulations` simulações independentes, cada uma com seu
        próprio ambiente sorteado do zero, e concatena as amostras.
        Garante que wind_direction (e demais variáveis) tenham variância
        real no dataset final.
        """

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
