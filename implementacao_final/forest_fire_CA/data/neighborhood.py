"""
Utilitário compartilhado para extrair a vizinhança de Moore de uma
célula em ordem fixa, usado tanto na geração do dataset de treino
(dataset_builder.py) quanto na inferência do CA (simulation/automata.py).

É crítico que ambos os lados usem exatamente a mesma ordem e a mesma
convenção de padding — qualquer divergência faria o Transformer
receber features fora da distribuição que ele aprendeu, sem erro
explícito (mesmo problema de categoria do bug do StandardScaler
visto anteriormente neste projeto).
"""

import numpy as np

# Ordem fixa das 8 direções (sentido horário a partir do Norte).
# Índice 8 (fora desta lista) é sempre a célula central.
NEIGHBOR_OFFSETS = [
    (-1,  0),  # N
    (-1,  1),  # NE
    ( 0,  1),  # E
    ( 1,  1),  # SE
    ( 1,  0),  # S
    ( 1, -1),  # SW
    ( 0, -1),  # W
    (-1, -1),  # NW
]

SEQ_LEN = len(NEIGHBOR_OFFSETS) + 1   # 8 vizinhos + célula central = 9

FEATURE_NAMES = [
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "elevation",
    "slope",
    "vegetation",
    "current_state",
]

N_FEATURES = len(FEATURE_NAMES)


def extract_cell_features(environment, grid, i, j):
    """
    Vetor de features de UMA célula (i,j). Usado tanto para a célula
    central quanto para cada vizinho.

    Se (i,j) estiver fora da grid, retorna um vetor de zeros (padding) —
    representa "não há célula aqui" (ex: borda da grid).
    """

    rows, cols = grid.shape

    if not (0 <= i < rows and 0 <= j < cols):
        return np.zeros(N_FEATURES, dtype=np.float32)

    return np.array([
        environment["temperature"][i, j],
        environment["humidity"][i, j],
        environment["wind_speed"][i, j],
        environment["wind_direction"][i, j],
        environment["elevation"][i, j],
        environment["slope"][i, j],
        float(environment["vegetation"][i, j]),
        float(grid[i, j]),
    ], dtype=np.float32)


def extract_neighborhood_sequence(environment, grid, i, j):
    """
    Monta a sequência (seq_len=9, n_features=8) para a célula (i,j):
    posições 0-7 = vizinhos na ordem fixa de NEIGHBOR_OFFSETS,
    posição 8    = célula central.

    Vizinhos fora da grid (borda) entram como vetor de zeros.
    """

    sequence = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)

    for idx, (di, dj) in enumerate(NEIGHBOR_OFFSETS):
        sequence[idx] = extract_cell_features(environment, grid, i + di, j + dj)

    sequence[-1] = extract_cell_features(environment, grid, i, j)

    return sequence
