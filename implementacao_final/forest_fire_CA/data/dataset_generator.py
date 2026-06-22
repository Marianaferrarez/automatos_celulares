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

    def generate_flammable_mask(
        self,
        elevation,
        protected_center_radius=3,
    ):

        raw_noise = np.random.rand(self.rows, self.cols)

        kernel_size = max(3, min(self.rows, self.cols) // 8)
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

        smooth_noise = self._smooth(raw_noise, kernel)

        non_flammable = smooth_noise > np.quantile(smooth_noise, 0.85)

        flammable = ~non_flammable

        center_i, center_j = self.rows // 2, self.cols // 2
        r = protected_center_radius
        i_lo, i_hi = max(0, center_i - r), min(self.rows, center_i + r + 1)
        j_lo, j_hi = max(0, center_j - r), min(self.cols, center_j + r + 1)
        flammable[i_lo:i_hi, j_lo:j_hi] = True

        return flammable

    @staticmethod
    def _smooth(field, kernel):
        rows, cols = field.shape
        kh, kw = kernel.shape

        pad_rows = rows + kh - 1
        pad_cols = cols + kw - 1

        field_f = np.fft.rfft2(field, s=(pad_rows, pad_cols))
        kernel_f = np.fft.rfft2(kernel, s=(pad_rows, pad_cols))

        conv = np.fft.irfft2(field_f * kernel_f, s=(pad_rows, pad_cols))

        start_r = (kh - 1) // 2
        start_c = (kw - 1) // 2

        return conv[start_r:start_r + rows, start_c:start_c + cols]
    
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

        flammable_mask = (
            self.generate_flammable_mask(
                elevation
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
            "flammable_mask": flammable_mask,
        }