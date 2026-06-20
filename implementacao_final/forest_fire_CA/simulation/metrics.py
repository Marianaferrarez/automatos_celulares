"""
Métricas de avaliação da simulação CA (Seção 2.5 do artigo).

Compara a grid simulada com uma grid de referência (ground truth),
ambas com estados {1=TREE, 2=BURNING, 3=BURNED}.

O artigo usa:
  - Accuracy  (eq. 8)
  - Kappa coefficient
  - IoU       (eq. 9)
"""

import numpy as np
from sklearn.metrics import cohen_kappa_score


def evaluate(simulated: np.ndarray, reference: np.ndarray) -> dict:
    """
    Parâmetros
    ----------
    simulated : np.ndarray (rows, cols) com valores {1, 2, 3}
    reference : np.ndarray (rows, cols) com valores {1, 2, 3}

    Retorna
    -------
    dict com accuracy, kappa e iou_burned

    Nota: esta métrica compara apenas o ESTADO FINAL das duas grids
    (assim como a Seção 2.5.3 do artigo). Em grids pequenas e sem
    grandes áreas de TREE isoladas por NONFLAMMABLE, duas simulações
    com regras de propagação diferentes tendem a convergir para o
    mesmo conjunto de células queimadas — o fogo, dado tempo
    suficiente, acaba alcançando tudo que é alcançável de qualquer
    jeito. Nesses casos, accuracy/kappa/IoU ficam artificialmente
    próximos de 1.0 mesmo quando os modelos discordam bastante sobre
    COMO o fogo se espalha. Use `evaluate_trajectory()` para uma
    comparação que enxergue essa diferença.
    """

    sim_flat = simulated.flatten()
    ref_flat = reference.flatten()

    # Accuracy geral (eq. 8)
    accuracy = float(np.mean(sim_flat == ref_flat))

    # Kappa
    kappa = cohen_kappa_score(ref_flat, sim_flat)

    # IoU da área queimada/queimando  (eq. 9)
    # Considera BURNING (2) e BURNED (3) como "área afetada"
    sim_fire = (sim_flat >= 2)
    ref_fire = (ref_flat >= 2)

    intersection = np.sum(sim_fire & ref_fire)
    union = np.sum(sim_fire | ref_fire)

    iou = float(intersection / union) if union > 0 else 0.0

    return {
        "accuracy": accuracy,
        "kappa": kappa,
        "iou": iou,
    }


def _ignition_step_map(frames, burning_value=2, burned_value=3):
    """
    Para cada célula, retorna o índice do PRIMEIRO passo em que ela
    aparece como BURNING ou BURNED. Células que nunca pegam fogo
    recebem -1.

    `frames` é uma lista de grids (um por passo da simulação).
    """
    rows, cols = frames[0].shape
    ignition_step = np.full((rows, cols), -1, dtype=int)

    for step, grid in enumerate(frames):
        on_fire = (grid == burning_value) | (grid == burned_value)
        newly_set = on_fire & (ignition_step == -1)
        ignition_step[newly_set] = step

    return ignition_step


def evaluate_trajectory(frames_simulated, frames_reference) -> dict:
    """
    Compara a TRAJETÓRIA de propagação entre duas simulações, não só
    o estado final. Para cada célula, mede em qual passo ela pegou
    fogo em cada simulação, e calcula:

      - mae_ignition_step : erro médio absoluto (em nº de passos) do
        instante de ignição, considerando só células que pegaram
        fogo em AMBAS as simulações.
      - frac_same_outcome : fração de células cujo destino final
        (pegou fogo ou não) é igual nas duas simulações — equivale
        à `accuracy` de evaluate(), mas reaproveitado aqui para
        contexto.
      - frac_only_simulated / frac_only_reference : fração de
        células que pegaram fogo em uma simulação mas não na outra
        (mede divergência genuína de alcance, não só de timing).

    Use isto junto com `evaluate()`: se accuracy/kappa/IoU do estado
    final estiverem muito altos mas `mae_ignition_step` for grande,
    os dois modelos chegam ao mesmo lugar por caminhos bem diferentes.
    """

    ign_sim = _ignition_step_map(frames_simulated)
    ign_ref = _ignition_step_map(frames_reference)

    caught_both = (ign_sim != -1) & (ign_ref != -1)
    caught_sim_only = (ign_sim != -1) & (ign_ref == -1)
    caught_ref_only = (ign_sim == -1) & (ign_ref != -1)

    total_cells = ign_sim.size

    if caught_both.sum() > 0:
        mae_ignition_step = float(
            np.mean(np.abs(ign_sim[caught_both] - ign_ref[caught_both]))
        )
    else:
        mae_ignition_step = float("nan")

    same_outcome = (ign_sim != -1) == (ign_ref != -1)
    frac_same_outcome = float(np.mean(same_outcome))

    return {
        "mae_ignition_step": mae_ignition_step,
        "frac_same_outcome": frac_same_outcome,
        "frac_only_simulated": float(caught_sim_only.sum() / total_cells),
        "frac_only_reference": float(caught_ref_only.sum() / total_cells),
        "n_cells_compared": int(caught_both.sum()),
    }


def print_metrics(metrics: dict, label: str = "Simulação CA") -> None:

    print(f"\n--- {label} ---")

    if "accuracy" in metrics:
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Kappa    : {metrics['kappa']:.4f}")
        print(f"  IoU      : {metrics['iou']:.4f}")

    if "mae_ignition_step" in metrics:
        print(f"  MAE do instante de ignição : {metrics['mae_ignition_step']:.2f} passos")
        print(f"  Mesmo destino final (%)    : {100*metrics['frac_same_outcome']:.2f}%")
        print(f"  Só pegou fogo no simulado  : {100*metrics['frac_only_simulated']:.2f}%")
        print(f"  Só pegou fogo na referência: {100*metrics['frac_only_reference']:.2f}%")
