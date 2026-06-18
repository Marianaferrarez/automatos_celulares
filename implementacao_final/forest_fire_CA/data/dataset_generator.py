import numpy as np


class DatasetGenerator:

    def __init__(self, rows=100, cols=100):

        self.rows = rows
        self.cols = cols

    def generate_temperature(
        self,
        elevation
    ):

        base_temp = 35

        temperature = (
            base_temp
            - elevation / 200
        )

        noise = np.random.normal(
            0,
            2,
            size=(self.rows, self.cols)
        )

        temperature += noise

        return np.clip(
            temperature,
            10,
            45
        )

    def generate_humidity(
        self,
        temperature
    ):

        humidity = (
            100
            - temperature * 2
        )

        humidity += np.random.normal(
            0,
            5,
            size=(self.rows, self.cols)
        )

        return np.clip(
            humidity,
            10,
            100
        )

    def generate_wind_speed(self):

        return np.random.uniform(
            low=0,
            high=25,
            size=(self.rows, self.cols)
        )

    def generate_vegetation(
        self,
        humidity
    ):

        probability = humidity / 100

        return (
            np.random.rand(
                self.rows,
                self.cols
            )
            < probability
        ).astype(int)
    
    def generate_elevation(self):

        x = np.linspace(-1, 1, self.cols)
        y = np.linspace(-1, 1, self.rows)

        X, Y = np.meshgrid(x, y)

        mountain = np.exp(
            -(X**2 + Y**2) * 4
        )

        elevation = (
            mountain * 2000
        )

        noise = np.random.normal(
            0,
            50,
            size=(self.rows, self.cols)
        )

        elevation += noise

        return np.clip(
            elevation,
            0,
            2000
        )

    def generate_slope(
        self,
        elevation
    ):

        gy, gx = np.gradient(
            elevation
        )

        slope = np.sqrt(
            gx**2 + gy**2
        )

        slope = (
            slope /
            slope.max()
        ) * 45

        return slope

    def generate_wind_direction(self):

        direction = np.random.uniform(
            0,
            360
        )

        return np.full(
            (self.rows, self.cols),
            direction
        )

    def generate_all(self):

        elevation = self.generate_elevation()

        slope = self.generate_slope(
            elevation
        )

        temperature = (
            self.generate_temperature(
                elevation
            )
        )

        humidity = (
            self.generate_humidity(
                temperature
            )
        )

        vegetation = (
            self.generate_vegetation(
                humidity
            )
        )

        return {
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed":
                self.generate_wind_speed(),
            "wind_direction":
                self.generate_wind_direction(),
            "elevation": elevation,
            "slope": slope,
            "vegetation": vegetation,
        }