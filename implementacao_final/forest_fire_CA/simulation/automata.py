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
        self.grid = np.ones((rows, cols), dtype=int)

        num_initial_fires = 50

        for _ in range(num_initial_fires):

            i = np.random.randint(0, rows)
            j = np.random.randint(0, cols)

            self.grid[i, j] = BURNING

    def neighbors(self, i, j):

        neigh = []

        for di in [-1,0,1]:
            for dj in [-1,0,1]:

                if di == 0 and dj == 0:
                    continue

                ni = i + di
                nj = j + dj

                if 0 <= ni < self.rows and 0 <= nj < self.cols:
                    neigh.append((ni,nj))

        return neigh

    def step(self):

        new_grid = self.grid.copy()

        for i in range(self.rows):
            for j in range(self.cols):

                state = self.grid[i,j]

                if state == BURNING:
                    new_grid[i,j] = BURNED

                elif state == TREE:

                    burning_neighbor = False

                    for ni,nj in self.neighbors(i,j):
                        if self.grid[ni,nj] == BURNING:
                            burning_neighbor = True
                            break

                    if burning_neighbor:
                        prob = self.ignition_probability(i, j)
                        if np.random.rand() < prob:
                            new_grid[i,j] = BURNING

        self.grid = new_grid
        
        
    def ignition_probability(self, i, j):
    
        temp = self.environment["temperature"][i, j]
        humidity = self.environment["humidity"][i, j]
        wind = self.environment["wind_speed"][i, j]
        vegetation = self.environment["vegetation"][i, j]
        elevation = self.environment["elevation"][i, j]
        slope = self.environment["slope"][i, j]

        score = (
            0.05 * temp
            -0.005 * humidity
            +0.04 * wind
            +0.30 * vegetation
            +0.0002 * elevation
            +0.02 * slope
        )

        return np.clip(score, 0, 1)