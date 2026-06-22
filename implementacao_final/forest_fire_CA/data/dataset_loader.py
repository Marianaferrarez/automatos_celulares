import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data.neighborhood import FEATURE_NAMES, SEQ_LEN, N_FEATURES


def _sequence_columns():
    cols = []
    for pos in range(SEQ_LEN):
        for fname in FEATURE_NAMES:
            cols.append(f"pos{pos}_{fname}")
    return cols

SEQUENCE_COLS = _sequence_columns()

class DatasetLoader:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def _resolve_csv_path(self):
        candidates = [
            Path(self.csv_path),
            Path(__file__).resolve().parents[1] / self.csv_path,
            Path(__file__).resolve().parents[3] / self.csv_path,
        ]

        for p in candidates:
            if p.exists():
                return p

        return Path(self.csv_path)

    def load_data(self):
        df = pd.read_csv(self._resolve_csv_path())

        df = df[df["next_state"].isin([1, 2, 3])].copy()

        X_flat = df[SEQUENCE_COLS].values.astype(np.float32)
        y = df["next_state"] - 1

        n_samples = X_flat.shape[0]
        X_seq = X_flat.reshape(n_samples, SEQ_LEN, N_FEATURES)

        X_for_scaler = X_seq.reshape(-1, N_FEATURES)

        scaler = StandardScaler()
        X_scaled_flat = scaler.fit_transform(X_for_scaler)

        X_scaled = X_scaled_flat.reshape(n_samples, SEQ_LEN, N_FEATURES)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        return X_train, X_test, y_train, y_test, scaler
