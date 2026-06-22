import numpy as np

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

SEQ_LEN = len(NEIGHBOR_OFFSETS) + 1

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
    sequence = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)

    for idx, (di, dj) in enumerate(NEIGHBOR_OFFSETS):
        sequence[idx] = extract_cell_features(environment, grid, i + di, j + dj)

    sequence[-1] = extract_cell_features(environment, grid, i, j)

    return sequence
