import pandas as pd
from pathlib import Path

from simulation.automata import ForestCA
from data.dataset_generator import DatasetGenerator
from visualization.fire_animation import FireAnimation

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

class DatasetBuilder:
    
    def __init__(
        self,
        rows=100,
        cols=100,
        steps=30
    ):

        self.rows = rows
        self.cols = cols
        self.steps = steps
        
    def build(self):
        generator = DatasetGenerator(
            self.rows,
            self.cols
        )

        environment = generator.generate_all()
        
        ca = ForestCA(
            environment=environment,
            rows=self.rows,
            cols=self.cols
        )
        samples = []
        
        for _ in range(self.steps):
        
            current_grid = ca.grid.copy()

            ca.step()

            next_grid = ca.grid.copy()
            
            for i in range(self.rows):
                for j in range(self.cols):
                    sample = {
                        "temperature":
                            environment["temperature"][i,j],
                        "humidity":
                            environment["humidity"][i,j],
                        "wind_speed":
                            environment["wind_speed"][i,j],
                        "wind_direction":
                            environment["wind_direction"][i,j],
                        "elevation":
                            environment["elevation"][i,j],
                        "slope":
                            environment["slope"][i,j],
                        "vegetation":
                            environment["vegetation"][i,j],
                        "current_state":
                            current_grid[i,j],
                        "next_state":
                            next_grid[i,j]
                    }
                    samples.append(sample)
                
        df = pd.DataFrame(samples)

        return df

    def build_excluding_burned(self):
        generator = DatasetGenerator(
            self.rows,
            self.cols
        )

        environment = generator.generate_all()

        ca = ForestCA(
            environment=environment,
            rows=self.rows,
            cols=self.cols
        )

        samples = []

        for _ in range(self.steps):

            current_grid = ca.grid.copy()

            ca.step()

            next_grid = ca.grid.copy()

            for i in range(self.rows):
                for j in range(self.cols):

                    if current_grid[i, j] == 3:
                        continue

                    sample = {
                        "temperature":
                            environment["temperature"][i, j],
                        "humidity":
                            environment["humidity"][i, j],
                        "wind_speed":
                            environment["wind_speed"][i, j],
                        "wind_direction":
                            environment["wind_direction"][i, j],
                        "elevation":
                            environment["elevation"][i, j],
                        "slope":
                            environment["slope"][i, j],
                        "vegetation":
                            environment["vegetation"][i, j],
                        "current_state":
                            current_grid[i, j],
                        "next_state":
                            next_grid[i, j]
                    }
                    samples.append(sample)

        df = pd.DataFrame(samples)

        return df
    
    def save_csv(
        self,
        filename="forest_fire_dataset.csv"
    ):

        df = self.build()

        df.to_csv(
            filename,
            index=False
        )

        print(
            f"Dataset salvo em {filename}"
        )

    def preview_animation(
        self,
        steps=50
    ):

        generator = DatasetGenerator(
            self.rows,
            self.cols
        )

        environment = generator.generate_all()

        ca = ForestCA(
            environment=environment,
            rows=self.rows,
            cols=self.cols
        )

        frames = []

        for _ in range(steps):

            frames.append(
                ca.grid.copy()
            )

            ca.step()

        FireAnimation.animate(
            frames
        )

    def save_csv_excluding_burned(
        self,
        filename="forest_fire_dataset_excluding_burned.csv"
    ):

        df = self.build_excluding_burned()

        df.to_csv(
            filename,
            index=False
        )

        print(
            f"Dataset experimental salvo em {filename}"
        )