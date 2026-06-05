import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pathlib import Path

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from data.dataset_loader import DatasetLoader
from data.dataset_builder import DatasetBuilder
from models.transformer import FireTransformer
from visualization.loss_plot import LossPlot
from visualization.confusion_matrix_plot import ConfusionMatrixPlot


BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.001
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CLASS_NAMES = ["TREE", "BURNING", "BURNED"]


loader = DatasetLoader(
    "forest_fire_dataset.csv"
)

# If the dataset CSV is missing in the expected locations, generate it
resolved = loader._resolve_csv_path()
if not resolved.exists():
    builder = DatasetBuilder(rows=40, cols=40, steps=15)
    df = builder.build()
    df.to_csv(resolved, index=False)

X_train, X_test, y_train, y_test, scaler = (
    loader.load_data()
)


X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train.values,
    dtype=torch.long
)

y_test = torch.tensor(
    y_test.values,
    dtype=torch.long
)

class_counts = torch.bincount(
    y_train,
    minlength=3
).float()

class_weights = class_counts.sum() / (
    len(class_counts) * class_counts
)

class_weights = torch.where(
    class_counts > 0,
    class_weights,
    torch.zeros_like(class_weights)
)

train_dataset = TensorDataset(
    X_train,
    y_train
)

test_dataset = TensorDataset(
    X_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE
)

model = FireTransformer(num_classes=3)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

losses = []
for epoch in range(EPOCHS):
    model.train()

    total_loss = 0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss: {avg_loss:.4f}"
    )
    
    model.eval()

    correct = 0
    total = 0
    
    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            outputs = model(X_batch)

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == y_batch
            ).sum().item()

            total += y_batch.size(0)

    accuracy = 100 * correct / total

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

LossPlot.plot(losses)

all_predictions = []
all_targets = []

model.eval()

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        outputs = model(X_batch)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_targets.extend(
            y_batch.cpu().numpy()
        )

cm = confusion_matrix(
    all_targets,
    all_predictions,
    labels=[0, 1, 2]
)

ConfusionMatrixPlot.plot(cm)

print(cm)

report = classification_report(
    all_targets,
    all_predictions,
    labels=[0, 1, 2],
    target_names=CLASS_NAMES,
    zero_division=0
)

print(
    report
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

(RESULTS_DIR / "classification_report.txt").write_text(
    report,
    encoding="utf-8"
)


