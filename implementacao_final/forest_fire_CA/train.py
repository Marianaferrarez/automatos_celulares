import pickle
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
)
from pathlib import Path

from torch.utils.data import TensorDataset, DataLoader

from data.dataset_loader import DatasetLoader
from data.dataset_builder import DatasetBuilder
from models.transformer import FireTransformer
from visualization.loss_plot import LossPlot
from visualization.confusion_matrix_plot import ConfusionMatrixPlot


BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SAVED_DIR = Path(__file__).resolve().parent / "saved_models"
CLASS_NAMES = ["TREE", "BURNING", "BURNED"]


loader = DatasetLoader("forest_fire_dataset.csv")

resolved = loader._resolve_csv_path()

_needs_rebuild = not resolved.exists()

if resolved.exists():
    # Checa se o CSV é do schema antigo (vizinhança não incluída).
    # Isso acontece se o usuário já tinha um dataset gerado antes da
    # adição da vizinhança de Moore como entrada do Transformer.
    import pandas as _pd
    _header = _pd.read_csv(resolved, nrows=0).columns.tolist()
    if "pos0_temperature" not in _header:
        print(
            "Aviso: forest_fire_dataset.csv está no formato antigo "
            "(sem vizinhança de Moore). Regerando dataset..."
        )
        _needs_rebuild = True

if _needs_rebuild:
    builder = DatasetBuilder(rows=40, cols=40, steps=50, n_simulations=8)
    df = builder.build()
    df.to_csv(resolved, index=False)

X_train, X_test, y_train, y_test, scaler = loader.load_data()

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.long)
y_test_t = torch.tensor(y_test.values, dtype=torch.long)

class_counts = torch.bincount(y_train_t, minlength=3).float()
class_weights = class_counts.sum() / (len(class_counts) * class_counts)
class_weights = torch.where(
    class_counts > 0,
    class_weights,
    torch.zeros_like(class_weights),
)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
test_loader = DataLoader(
    TensorDataset(X_test_t, y_test_t),
    batch_size=BATCH_SIZE,
)

model = FireTransformer(num_classes=3)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

losses = []
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}  "
        f"Loss: {avg_loss:.4f}  "
        f"Acc: {100 * correct / total:.2f}%"
    )

LossPlot.plot(losses)

# ------------------------------------------------------------------ #
# Avaliação final                                                     #
# ------------------------------------------------------------------ #
all_preds, all_targets = [], []

model.eval()
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())

cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
ConfusionMatrixPlot.plot(cm)
print(cm)

report = classification_report(
    all_targets,
    all_preds,
    labels=[0, 1, 2],
    target_names=CLASS_NAMES,
    zero_division=0,
)
print(report)

# Kappa e IoU (métricas do artigo)
kappa = cohen_kappa_score(all_targets, all_preds)
print(f"Kappa: {kappa:.4f}")

# IoU para a classe BURNING (índice 1)
tp = cm[1, 1]
fp = cm[0, 1] + cm[2, 1]
fn = cm[1, 0] + cm[1, 2]
iou_burning = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
print(f"IoU (BURNING): {iou_burning:.4f}")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

metrics_text = (
    report
    + f"\nKappa: {kappa:.4f}\n"
    + f"IoU (BURNING): {iou_burning:.4f}\n"
)
(RESULTS_DIR / "classification_report.txt").write_text(
    metrics_text, encoding="utf-8"
)

# ------------------------------------------------------------------ #
# Salva modelo + scaler (necessário para o CA usar o Transformer)     #
# ------------------------------------------------------------------ #
SAVED_DIR.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), SAVED_DIR / "fire_transformer.pth")

with open(SAVED_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"\nModelo salvo em {SAVED_DIR / 'fire_transformer.pth'}")
print(f"Scaler salvo em {SAVED_DIR / 'scaler.pkl'}")
