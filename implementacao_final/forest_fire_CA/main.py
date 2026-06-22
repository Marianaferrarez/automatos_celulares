import pickle
import subprocess
import sys
from pathlib import Path

import torch

from data.dataset_builder import DatasetBuilder
from models.transformer import FireTransformer
from simulation.automata import ForestCA
from data.dataset_generator import DatasetGenerator
from simulation.metrics import evaluate, evaluate_trajectory, print_metrics
from visualization.fire_animation import FireAnimation
from visualization.state_distribution import StateDistribution
from visualization.grid_comparison import GridComparison

ROOT_DIR = Path(__file__).resolve().parent
SAVED_DIR = ROOT_DIR / "saved_models"

ROWS = 40
COLS = 40
STEPS = 50
ANIMATION_FRAMES = 50
THRESHOLD = 0.20
ALPHA = 0.3
N_SIMULATIONS = 8

def _build_dataset():
    csv_path = ROOT_DIR / "forest_fire_dataset.csv"

    needs_rebuild = not csv_path.exists()

    if csv_path.exists():
        import pandas as pd
        header = pd.read_csv(csv_path, nrows=0).columns.tolist()
        if "pos0_temperature" not in header:
            print(
                "Aviso: forest_fire_dataset.csv está no formato antigo "
                "(sem vizinhança de Moore). Regerando dataset..."
            )
            needs_rebuild = True

    if needs_rebuild:
        print("Gerando dataset...")
        builder = DatasetBuilder(
            rows=ROWS,
            cols=COLS,
            steps=STEPS,
            n_simulations=N_SIMULATIONS,
        )
        df = builder.build()
        df.to_csv(csv_path, index=False)
        StateDistribution.plot(df)

    return csv_path

def _train_model():
    print("\nTreinando Transformer...")
    subprocess.run(
        [sys.executable, str(ROOT_DIR / "train.py")],
        cwd=ROOT_DIR,
        check=True,
    )

def _load_model():
    model_path = SAVED_DIR / "fire_transformer.pth"
    scaler_path = SAVED_DIR / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        print("Aviso: modelo treinado não encontrado. CA usará fórmula analítica.")
        return None, None

    model = FireTransformer(num_classes=3)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    print("Transformer carregado com sucesso.")
    return model, scaler

def _run_to_completion(environment, model, scaler, max_steps):
    ca = ForestCA(
        environment=environment,
        rows=ROWS,
        cols=COLS,
        model=model,
        scaler=scaler,
        alpha=ALPHA,
        threshold=THRESHOLD,
    )

    frames = [ca.grid.copy()]
    for _ in range(max_steps):
        ca.step()
        frames.append(ca.grid.copy())
        if int((ca.grid == 2).sum()) == 0:
            break

    return frames, ca.grid


def _run_simulation(model, scaler):
    print("\nRodando simulação CA com Transformer...")

    generator = DatasetGenerator(ROWS, COLS)
    environment = generator.generate_all()

    frames_transformer, grid_transformer = _run_to_completion(
        environment, model=model, scaler=scaler, max_steps=ANIMATION_FRAMES
    )

    for step, frame in enumerate(frames_transformer):
        burning = int((frame == 2).sum())
        burned = int((frame == 3).sum())
        print(
            f"  Passo {step:2d}/{ANIMATION_FRAMES}"
            f"  queimando={burning:4d}"
            f"  queimado={burned:4d}"
        )

    while len(frames_transformer) < ANIMATION_FRAMES:
        frames_transformer.append(frames_transformer[-1].copy())

    FireAnimation.animate(frames_transformer)
    print("Animação salva em results/fire_spread.gif")

    print("\nRodando simulação CA de referência (fórmula analítica)...")
    frames_analytical, grid_analytical = _run_to_completion(
        environment, model=None, scaler=None, max_steps=ANIMATION_FRAMES
    )

    metrics = evaluate(simulated=grid_transformer, reference=grid_analytical)
    print_metrics(metrics, label="Transformer-CA vs Analítica-CA (estado final)")

    metrics_traj = evaluate_trajectory(frames_transformer, frames_analytical)
    print_metrics(metrics_traj, label="Transformer-CA vs Analítica-CA (trajetória)")

    GridComparison.plot(
        simulated=grid_transformer,
        reference=grid_analytical,
        metrics=metrics,
        title="Transformer-CA vs Analítica-CA",
    )
    print("Comparação salva em results/grid_comparison.png")


def main():
    _build_dataset()
    _train_model()

    model, scaler = _load_model()
    _run_simulation(model, scaler)

if __name__ == "__main__":
    main()
