import numpy as np
from sklearn.metrics import cohen_kappa_score


def evaluate(simulated: np.ndarray, reference: np.ndarray) -> dict:
    sim_flat = simulated.flatten()
    ref_flat = reference.flatten()

    accuracy = float(np.mean(sim_flat == ref_flat))

    kappa = cohen_kappa_score(ref_flat, sim_flat)

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
    rows, cols = frames[0].shape
    ignition_step = np.full((rows, cols), -1, dtype=int)

    for step, grid in enumerate(frames):
        on_fire = (grid == burning_value) | (grid == burned_value)
        newly_set = on_fire & (ignition_step == -1)
        ignition_step[newly_set] = step

    return ignition_step


def evaluate_trajectory(frames_simulated, frames_reference) -> dict:
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
