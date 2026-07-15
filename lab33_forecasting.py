import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time

# ============================================================
# HYPERPARAMETERS
# ============================================================

SEQ_LEN = 30
PRED_LEN = 1

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# GENERATE SYNTHETIC TIME SERIES
# ============================================================

np.random.seed(42)

t = np.arange(0, 1000)

series = (
    np.sin(0.02 * t)
    + 0.5 * np.sin(0.05 * t)
    + 0.1 * np.random.randn(len(t))
)

# ============================================================
# DATASET
# ============================================================

class TimeSeriesDataset(Dataset):

    def __init__(self, series, seq_len):

        self.series = series
        self.seq_len = seq_len

    def __len__(self):

        return len(self.series) - self.seq_len

    def __getitem__(self, idx):

        x = self.series[idx : idx + self.seq_len]

        y = self.series[idx + self.seq_len]

        x = torch.tensor(
            x,
            dtype=torch.float32
        ).unsqueeze(-1)

        y = torch.tensor(
            y,
            dtype=torch.float32
        )

        return x, y

# ============================================================
# SPLIT DATA
# ============================================================

train_size = int(0.8 * len(series))

train_series = series[:train_size]
val_series = series[train_size:]

train_ds = TimeSeriesDataset(
    train_series,
    SEQ_LEN
)

val_ds = TimeSeriesDataset(
    val_series,
    SEQ_LEN
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE
)

# ============================================================
# BASELINE MODEL (NAIVE)
# ============================================================

def naive_forecast(x):

    return x[:, -1, 0]

# ============================================================
# CNN FORECASTER
# ============================================================

class CNNForecaster(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):

        # (B,SEQ,1) -> (B,1,SEQ)
        x = x.permute(0, 2, 1)

        x = self.conv(x)

        x = x.squeeze(-1)

        x = self.fc(x)

        return x.squeeze(-1)

# ============================================================
# TRANSFORMER FORECASTER
# ============================================================

class TransformerForecaster(nn.Module):

    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_layers=2
    ):

        super().__init__()

        self.input_proj = nn.Linear(
            1,
            d_model
        )

        self.pos_emb = nn.Embedding(
            SEQ_LEN,
            d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(
            d_model,
            1
        )

    def forward(self, x):

        B, T, _ = x.shape

        pos = torch.arange(
            T,
            device=x.device
        )

        x = self.input_proj(x)

        x = x + self.pos_emb(pos).unsqueeze(0)

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.fc(x)

        return x.squeeze(-1)

# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=EPOCHS
):

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    model.to(DEVICE)

    start = time.time()

    for epoch in range(1, epochs + 1):

        model.train()

        train_loss = 0

        for x, y in train_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            pred = model(x)

            loss = criterion(pred, y)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        # VALIDATION
        model.eval()

        val_loss = 0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(DEVICE)
                y = y.to(DEVICE)

                pred = model(x)

                loss = criterion(pred, y)

                val_loss += loss.item()

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Val Loss: {val_loss / len(val_loader):.4f}"
        )

    elapsed = time.time() - start

    return elapsed

# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, loader):

    criterion = nn.MSELoss()

    model.eval()

    total_loss = 0

    preds = []
    targets = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)

            loss = criterion(pred, y)

            total_loss += loss.item()

            preds.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())

    mse = total_loss / len(loader)

    return mse, preds, targets

# ============================================================
# NAIVE BASELINE EVALUATION
# ============================================================

naive_preds = []
naive_targets = []

for x, y in val_loader:

    pred = naive_forecast(x)

    naive_preds.extend(pred.numpy())
    naive_targets.extend(y.numpy())

naive_mse = np.mean(
    (
        np.array(naive_preds)
        - np.array(naive_targets)
    ) ** 2
)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n===================================")
    print(" TIME SERIES FORECASTING LAB 3.3 ")
    print(" CNN vs TRANSFORMER ")
    print("===================================\n")

    # ========================================================
    # CNN
    # ========================================================

    print("\nTraining CNN Forecaster...\n")

    cnn_model = CNNForecaster()

    cnn_time = train_model(
        cnn_model,
        train_loader,
        val_loader
    )

    cnn_mse, cnn_preds, cnn_targets = evaluate_model(
        cnn_model,
        val_loader
    )

    print(f"\nCNN Validation MSE: {cnn_mse:.6f}")

    # ========================================================
    # TRANSFORMER
    # ========================================================

    print("\nTraining Transformer Forecaster...\n")

    trans_model = TransformerForecaster()

    trans_time = train_model(
        trans_model,
        train_loader,
        val_loader
    )

    trans_mse, trans_preds, trans_targets = evaluate_model(
        trans_model,
        val_loader
    )

    print(f"\nTransformer Validation MSE: {trans_mse:.6f}")

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n===================================")
    print(" FINAL COMPARISON ")
    print("===================================\n")

    print(f"Naive Baseline MSE : {naive_mse:.6f}")
    print(f"CNN MSE            : {cnn_mse:.6f}")
    print(f"Transformer MSE    : {trans_mse:.6f}")

    print(f"\nCNN Training Time         : {cnn_time:.1f} s")
    print(f"Transformer Training Time : {trans_time:.1f} s")

    # ========================================================
    # VISUALIZATION
    # ========================================================

    plt.figure(figsize=(12, 5))

    plt.plot(
        cnn_targets[:100],
        label="Ground Truth"
    )

    plt.plot(
        cnn_preds[:100],
        label="CNN Prediction"
    )

    plt.plot(
        trans_preds[:100],
        label="Transformer Prediction"
    )

    plt.legend()

    plt.title("Forecast Comparison")

    plt.xlabel("Time Step")

    plt.ylabel("Value")

    plt.savefig("output/forecast_comparison.png")
    plt.show()

    # ========================================================
    # BAR CHART
    # ========================================================

    models = [
        "Naive",
        "CNN",
        "Transformer"
    ]

    mse_values = [
        naive_mse,
        cnn_mse,
        trans_mse
    ]

    plt.figure(figsize=(6, 4))

    plt.bar(models, mse_values)

    plt.ylabel("Validation MSE")

    plt.title("Model Comparison")

    plt.savefig("output/model_comparison.png") 
    plt.show()