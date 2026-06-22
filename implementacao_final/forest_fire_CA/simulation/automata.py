import numpy as np

from data.neighborhood import extract_neighborhood_sequence

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

EMPTY = 0
TREE = 1
BURNING = 2
BURNED = 3
NONFLAMMABLE = EMPTY


class ForestCA:
    def __init__(
        self,
        environment,
        rows=100,
        cols=100,
        model=None,
        scaler=None,
        alpha=0.3,
        threshold=0.20,
    ):
        self.rows = rows
        self.cols = cols
        self.environment = environment
        self.model = model
        self.scaler = scaler
        self.alpha = alpha
        self.threshold = threshold

        self.grid = np.full((rows, cols), TREE, dtype=int)

        flammable_mask = self.environment.get("flammable_mask")
        if flammable_mask is not None:
            self.grid[~flammable_mask] = NONFLAMMABLE

        center_i = rows // 2
        center_j = cols // 2

        for ni, nj in self.neighbors(center_i, center_j):
            if self.grid[ni, nj] == NONFLAMMABLE:
                self.grid[ni, nj] = TREE

        self.grid[center_i, center_j] = BURNING

    def neighbors(self, i, j):
        neigh = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.rows and 0 <= nj < self.cols:
                    neigh.append((ni, nj))
        return neigh

    def _random_factor(self):
        gamma = np.random.uniform(1e-4, 1.0)
        return 1.0 + (-np.log(gamma)) ** self.alpha

    def _wind_factor(self, i, j, burning_i, burning_j):
        wind_speed     = self.environment["wind_speed"][i, j]
        wind_direction = self.environment["wind_direction"][i, j]

        spread_di = i - burning_i
        spread_dj = j - burning_j

        angle = (np.degrees(np.arctan2(spread_di, spread_dj)) + 360) % 360
        diff  = min(abs(angle - wind_direction), 360 - abs(angle - wind_direction))

        wind_alignment = np.cos(np.radians(diff))

        wind_factor = 0.5 + 0.5 * (wind_speed / 25.0) * wind_alignment

        return float(np.clip(wind_factor, 0.1, 1.0))

    def _local_prob_analytical(self, i, j):
        temperature = self.environment["temperature"][i, j]
        humidity    = self.environment["humidity"][i, j]
        vegetation  = self.environment["vegetation"][i, j]
        elevation   = self.environment["elevation"][i, j]
        slope       = self.environment["slope"][i, j]

        temp_factor      = (temperature - 15.0) / 30.0
        humidity_factor  = 1.0 - (humidity - 10.0) / 90.0
        vegetation_factor = float(vegetation)
        slope_factor      = slope / 45.0
        elevation_factor  = elevation / 2000.0

        score = (
            0.25 * temp_factor
            + 0.25 * humidity_factor
            + 0.30 * vegetation_factor
            + 0.15 * slope_factor
            + 0.05 * elevation_factor
        )
        return float(np.clip(score, 0.0, 1.0))

    def _local_prob_transformer(self, i, j):
        if not _TORCH_AVAILABLE:
            return self._local_prob_analytical(i, j)

        sequence = extract_neighborhood_sequence(
            self.environment, self.grid, i, j
        )

        if self.scaler is not None:
            sequence = self.scaler.transform(sequence)

            sequence = np.clip(sequence, -5.0, 5.0)

        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1)

        return float(probs[0, 1].item())

    def _local_prob(self, i, j):
        if self.model is not None:
            return self._local_prob_transformer(i, j)
        return self._local_prob_analytical(i, j)

    def step(self):
        new_grid = self.grid.copy()

        for i in range(self.rows):
            for j in range(self.cols):

                state = self.grid[i, j]

                if state == NONFLAMMABLE:
                    continue

                elif state == BURNING:
                    new_grid[i, j] = BURNED

                elif state == TREE:

                    burning_neighbors = [
                        (ni, nj)
                        for ni, nj in self.neighbors(i, j)
                        if self.grid[ni, nj] == BURNING
                    ]

                    if not burning_neighbors:
                        continue

                    ft = self._local_prob(i, j)

                    wt = max(
                        self._wind_factor(i, j, ni, nj)
                        for ni, nj in burning_neighbors
                    )

                    ra = self._random_factor()

                    pt = float(np.clip(ra * ft * wt, 0.0, 1.0))

                    if pt >= self.threshold:
                        new_grid[i, j] = BURNING

        self.grid = new_grid
