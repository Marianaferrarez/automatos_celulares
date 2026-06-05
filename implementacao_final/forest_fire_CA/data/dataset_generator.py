import numpy as np


class DatasetGenerator:

    def __init__(self, rows=100, cols=100):

        self.rows = rows
        self.cols = cols

    def generate_temperature(self):

        return np.random.uniform(
            low=15,
            high=45,
            size=(self.rows, self.cols)
        )

    def generate_humidity(self):

        return np.random.uniform(
            low=10,
            high=100,
            size=(self.rows, self.cols)
        )

    def generate_wind_speed(self):

        return np.random.uniform(
            low=0,
            high=25,
            size=(self.rows, self.cols)
        )

    def generate_vegetation(self):

        return np.random.choice(
            [0, 1],
            size=(self.rows, self.cols),
            p=[0.2, 0.8]
        )
    
    def generate_elevation(self):
    
        return np.random.uniform(
            low=0,
            high=2000,
            size=(self.rows, self.cols)
        )


    def generate_slope(self):

        return np.random.uniform(
            low=0,
            high=45,
            size=(self.rows, self.cols)
        )


    def generate_wind_direction(self):

        return np.random.uniform(
            low=0,
            high=360,
            size=(self.rows, self.cols)
        )

    def generate_all(self):

        wind_speed = self.generate_wind_speed()

        return {
            "temperature": self.generate_temperature(),
            "humidity": self.generate_humidity(),
            "wind_speed": wind_speed,
            "vegetation": self.generate_vegetation(),
            "wind_direction": self.generate_wind_direction(),
            "elevation": self.generate_elevation(),
            "slope": self.generate_slope(),
        }