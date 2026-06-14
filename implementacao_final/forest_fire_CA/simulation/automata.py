import numpy as np

EMPTY = 0
TREE = 1
BURNING = 2
BURNED = 3


class ForestCA:

    def __init__(
        self,
        environment,
        rows=100,
        cols=100
    ):
        self.rows = rows
        self.cols = cols
        self.environment = environment

        # Floresta inicial
        self.grid = np.full(
            (rows, cols),
            TREE,
            dtype=int
        )

        # Foco inicial no centro
        center_i = rows // 2
        center_j = cols // 2

        self.grid[
            center_i,
            center_j
        ] = BURNING

    def neighbors(self, i, j):

        neigh = []

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:

                if di == 0 and dj == 0:
                    continue

                ni = i + di
                nj = j + dj

                if (
                    0 <= ni < self.rows and
                    0 <= nj < self.cols
                ):
                    neigh.append((ni, nj))

        return neigh

    def step(self):

        new_grid = self.grid.copy()

        for i in range(self.rows):
            for j in range(self.cols):

                state = self.grid[i, j]

                if state == BURNING:

                    new_grid[i, j] = BURNED

                elif state == TREE:

                    burning_neighbors = []

                    for ni, nj in self.neighbors(i, j):

                        if self.grid[ni, nj] == BURNING:
                            burning_neighbors.append(
                                (ni, nj)
                            )

                    if len(burning_neighbors) == 0:
                        continue

                    max_prob = 0.0

                    for ni, nj in burning_neighbors:

                        prob = self.ignition_probability(
                            i,
                            j,
                            ni,
                            nj
                        )

                        max_prob = max(
                            max_prob,
                            prob
                        )

                    if np.random.rand() < max_prob:

                        new_grid[i, j] = BURNING

        self.grid = new_grid

    def ignition_probability(
        self,
        i,
        j,
        burning_i,
        burning_j
    ):

        temperature = self.environment[
            "temperature"
        ][i, j]

        humidity = self.environment[
            "humidity"
        ][i, j]

        wind_speed = self.environment[
            "wind_speed"
        ][i, j]

        wind_direction = self.environment[
            "wind_direction"
        ][i, j]

        vegetation = self.environment[
            "vegetation"
        ][i, j]

        elevation = self.environment[
            "elevation"
        ][i, j]

        slope = self.environment[
            "slope"
        ][i, j]

        # -------------------------
        # Fatores ambientais
        # -------------------------

        temp_factor = (
            (temperature - 15.0)
            / 30.0
        )

        humidity_factor = (
            1.0
            - (humidity - 10.0)
            / 90.0
        )

        vegetation_factor = vegetation

        slope_factor = slope / 45.0

        elevation_factor = elevation / 2000.0

        # -------------------------
        # Influência do vento
        # -------------------------

        spread_di = i - burning_i
        spread_dj = j - burning_j

        angle = np.degrees(
            np.arctan2(
                spread_di,
                spread_dj
            )
        )

        angle = (angle + 360) % 360

        diff = abs(
            angle - wind_direction
        )

        diff = min(
            diff,
            360 - diff
        )

        wind_alignment = np.cos(
            np.radians(diff)
        )

        wind_factor = (
            (wind_speed / 25.0)
            * wind_alignment
        )

        # -------------------------
        # Score final
        # -------------------------

        score = (
            0.20 * temp_factor
            + 0.20 * humidity_factor
            + 0.25 * vegetation_factor
            + 0.15 * slope_factor
            + 0.05 * elevation_factor
            + 0.15 * wind_factor
        )

        return np.clip(
            score,
            0.0,
            1.0
        )