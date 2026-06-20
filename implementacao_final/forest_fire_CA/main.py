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
THRESHOLD = 0.20   # calibrado para a escala de Pt = RA*Ft*Wt (ver automata.py)
ALPHA = 0.3        # parâmetro RA (eq. 6) — 0 = sem aleatoriedade extra
N_SIMULATIONS = 8  # nº de ambientes independentes p/ gerar o dataset


def _build_dataset():
    """
    Gera o CSV de treino se não existir, ou se existir num schema
    antigo (sem a vizinhança de Moore — colunas pos0_..pos8_).
    """
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
    """Executa train.py e salva modelo + scaler."""
    print("\nTreinando Transformer...")
    subprocess.run(
        [sys.executable, str(ROOT_DIR / "train.py")],
        cwd=ROOT_DIR,
        check=True,
    )


def _load_model():
    """
    Carrega FireTransformer e StandardScaler do disco.
    Retorna (model, scaler) ou (None, None) se os arquivos não existirem.
    """
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
    """
    Roda o CA até o fogo se apagar (ou max_steps), retornando os frames
    e a grid final. Usado tanto para a versão Transformer quanto para
    a versão analítica, sobre o MESMO ambiente — para que a comparação
    isole o efeito de quem gera Ft(i,j).
    """
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
    """
    Roda duas simulações no MESMO ambiente sintético:
      1) Transformer-CA  — Ft(i,j) vem do modelo treinado
      2) Analítica-CA    — Ft(i,j) vem da fórmula heurística (fallback)

    Isso espelha a comparação Transformer-CA vs LSSVM-CA do artigo
    (Seção 2.5), usando a versão analítica como referência, já que
    não há fogo real para comparar com dados sintéticos.
    """
    print("\nRodando simulação CA com Transformer...")

    generator = DatasetGenerator(ROWS, COLS)
    environment = generator.generate_all()

    # --- Simulação 1: Transformer-CA ---
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

    # Completa a animação até ANIMATION_FRAMES repetindo o último frame,
    # para o gif ter duração fixa mesmo que o fogo apague antes
    while len(frames_transformer) < ANIMATION_FRAMES:
        frames_transformer.append(frames_transformer[-1].copy())

    FireAnimation.animate(frames_transformer)
    print("Animação salva em results/fire_spread.gif")

    # --- Simulação 2: Analítica-CA (referência, mesmo ambiente) ---
    print("\nRodando simulação CA de referência (fórmula analítica)...")
    frames_analytical, grid_analytical = _run_to_completion(
        environment, model=None, scaler=None, max_steps=ANIMATION_FRAMES
    )

    # --- Comparação do ESTADO FINAL (equivalente à Seção 2.5.3 do artigo) ---
    metrics = evaluate(simulated=grid_transformer, reference=grid_analytical)
    print_metrics(metrics, label="Transformer-CA vs Analítica-CA (estado final)")

    # --- Comparação da TRAJETÓRIA (complementar) ---
    # Em grids pequenas sem grandes áreas isoladas, duas simulações
    # costumam convergir para o mesmo estado final mesmo usando regras
    # de propagação diferentes — o fogo, dado tempo, alcança tudo que
    # é alcançável de qualquer jeito. Por isso accuracy/kappa/IoU do
    # estado final sozinhos podem ficar perto de 1.0 mesmo quando os
    # modelos discordam bastante sobre COMO e QUANDO o fogo se espalha.
    # A métrica de trajetória mede essa diferença de timing/alcance.
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
