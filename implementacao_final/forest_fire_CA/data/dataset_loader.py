import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DatasetLoader:

    def __init__(self, csv_path):

        self.csv_path = csv_path

    def _resolve_csv_path(self):

        candidate_paths = [
            Path(self.csv_path),
            Path(__file__).resolve().parents[1] / self.csv_path,
            Path(__file__).resolve().parents[3] / self.csv_path,
        ]

        for candidate_path in candidate_paths:
            if candidate_path.exists():
                return candidate_path

        return Path(self.csv_path)
        
    def load_data(self):
    
        df = pd.read_csv(self._resolve_csv_path())

        X = df[
            [
                "temperature",
                "humidity",
                "wind_speed",
                "wind_direction",
                "elevation",
                "slope",
                "vegetation",
                "current_state"
            ]
        ]

        y = df["next_state"] - 1

        scaler = StandardScaler()

        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            scaler
        )