"""
Experimento do threshold de probabilidade de queima — replica a
análise da Seção 4.1 / Fig. 10-11 / Tabela 3 do artigo (Zhou et al.,
2025).

O artigo treina o modelo UMA vez e depois varia só o limiar de
ignição (eq. 1) na hora de rodar a simulação, plotando Accuracy,
Kappa e IoU contra o threshold. É importante não confundir esse
threshold com o `generation_threshold` usado em data/dataset_builder.py
(que serve só para gerar exemplos de treino variados) — aqui estamos
testando o Transformer JÁ TREINADO, fixo, variando apenas a regra de
decisão na Eq. 1 do artigo: "ignição se Pt >= threshold".

Para isolar o efeito do threshold (e não misturar com a aleatoriedade
de qual ambiente foi sorteado), este script:
  1. Gera UM único ambiente sintético e o fixa como cenário de teste.
  2. Gera UMA grid de referência fixa (a "queimada real" deste
     experimento), simulando com a fórmula analítica em um threshold
     de referência razoável (THRESHOLD_REFERENCE).
  3. Roda a simulação Transformer-CA repetidamente sobre o MESMO
     ambiente, variando o threshold de 0.10 a 0.95.
  4. Para cada threshold, compara o resultado do Transformer-CA contra
     a grid de referência fixa do passo 2, calculando Accuracy, Kappa
     e IoU.
  5. Plota os 3 valores contra o threshold (igual à Fig. 11) e salva
     a tabela em CSV (igual à Tabela 3).

Não altera nada do pipeline principal (main.py, train.py) — é um
script independente que reusa o modelo e o scaler já treinados e
salvos em saved_models/.

Uso:
    python threshold_experiment.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.dataset_generator import DatasetGenerator
from models.transformer import FireTransformer
from simulation.automata import ForestCA, BURNING
from simulation.metrics import evaluate, print_metrics


ROOT_DIR = Path(__file__).resolve().parent
SAVED_DIR = ROOT_DIR / "saved_models"
RESULTS_DIR = ROOT_DIR / "results"

ROWS = 40
COLS = 40
MAX_STEPS = 60
ALPHA = 0.3

# Threshold usado para gerar a grid de referência fixa deste
# experimento (o "fogo real" contra o qual comparamos). Um valor
# intermediário e estável, fora dos extremos que estamos testando.
THRESHOLD_REFERENCE = 0.30

# Faixa de thresholds testada — espelha a Tabela 3 do artigo (0.50 a
# 1.00), mas ajustada à escala de Pt deste código (ver discussão sobre
# o fator RA inflar Pt). Testamos uma faixa mais ampla para também
# cobrir a região onde o efeito é mais visível aqui.
THRESHOLDS = np.round(np.arange(0.10, 1.00, 0.05), 2)


def _load_model():

    model_path = SAVED_DIR / "fire_transformer.pth"
    scaler_path = SAVED_DIR / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Modelo treinado não encontrado em saved_models/. "
            "Rode main.py (ou train.py) pelo menos uma vez antes "
            "deste experimento."
        )

    model = FireTransformer(num_classes=3)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


def _run_to_completion(environment, model, scaler, threshold, max_steps=MAX_STEPS):

    ca = ForestCA(
        environment=environment,
        rows=ROWS,
        cols=COLS,
        model=model,
        scaler=scaler,
        alpha=ALPHA,
        threshold=threshold,
    )

    for _ in range(max_steps):
        ca.step()
        if int((ca.grid == BURNING).sum()) == 0:
            break

    return ca.grid


def run_experiment():

    print("Carregando Transformer treinado...")
    model, scaler = _load_model()

    print("Gerando ambiente sintético fixo para o experimento...")
    np.random.seed(2024)   # fixo para reprodutibilidade do experimento
    generator = DatasetGenerator(ROWS, COLS)
    environment = generator.generate_all()

    print(
        f"Gerando grid de referência (fórmula analítica, "
        f"threshold={THRESHOLD_REFERENCE})..."
    )
    np.random.seed(1)   # fixa o RA da referência também
    reference_grid = _run_to_completion(
        environment, model=None, scaler=None, threshold=THRESHOLD_REFERENCE
    )
    print(f"  Referência: {int((reference_grid == 3).sum())} células queimadas")

    rows = []

    print(f"\nTestando {len(THRESHOLDS)} valores de threshold no Transformer-CA...")
    for threshold in THRESHOLDS:

        np.random.seed(1)   # mesmo RA em todas as rodadas, só threshold varia
        sim_grid = _run_to_completion(
            environment, model=model, scaler=scaler, threshold=threshold
        )

        metrics = evaluate(simulated=sim_grid, reference=reference_grid)

        burned_pct = 100 * (sim_grid == 3).sum() / (ROWS * COLS)

        rows.append({
            "threshold": threshold,
            "accuracy": metrics["accuracy"],
            "kappa": metrics["kappa"],
            "iou": metrics["iou"],
            "burned_pct": burned_pct,
        })

        print(
            f"  threshold={threshold:.2f}  "
            f"accuracy={metrics['accuracy']:.3f}  "
            f"kappa={metrics['kappa']:.3f}  "
            f"iou={metrics['iou']:.3f}  "
            f"queimado={burned_pct:.1f}%"
        )

    df = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "threshold_experiment.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nTabela salva em {csv_path}")

    _plot(df)

    best_row = df.loc[df["kappa"].idxmax()]
    print(
        f"\nMelhor threshold (maior Kappa): {best_row['threshold']:.2f} "
        f"(Kappa={best_row['kappa']:.3f}, Accuracy={best_row['accuracy']:.3f}, "
        f"IoU={best_row['iou']:.3f})"
    )

    return df


def _plot(df):

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df["threshold"], df["accuracy"], marker="o", label="Accuracy")
    ax.plot(df["threshold"], df["kappa"], marker="s", label="Kappa")
    ax.plot(df["threshold"], df["iou"], marker="^", label="IoU")

    ax.axvline(
        THRESHOLD_REFERENCE,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Threshold da referência ({THRESHOLD_REFERENCE})",
    )

    ax.set_xlabel("Burning probability threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Impacto do threshold no Transformer-CA\n(equivalente à Fig. 11 do artigo)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    out_path = RESULTS_DIR / "threshold_experiment.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Gráfico salvo em {out_path}")


if __name__ == "__main__":
    run_experiment()
