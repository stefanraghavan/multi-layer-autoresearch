"""Neural network training for daily stock direction prediction.

Single-file PyTorch implementation. Binary classification: up or down at close.
Walk-forward validation (no random splits).

Architecture is a simple MLP — the focus is on feature research (Layer 1)
and hyperparameter tuning (Layer 3), not model complexity.

*** LAYER 3 MODIFIES THE HYPERPARAMETERS ***

Usage:
    python train.py                         # Train with defaults
    TICKER=_combined python train.py        # Train on combined dataset
    TIME_BUDGET=60 python train.py          # 60-second budget
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Constants (fixed evaluation protocol — do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = int(os.environ.get("TIME_BUDGET", "60"))  # seconds
EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "200"))  # epochs without val_loss improvement
TICKER = os.environ.get("TICKER", "_combined").lower()
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))
SEED = int(os.environ.get("SEED", "0"))  # 0 = no fixed seed

# ---------------------------------------------------------------------------
# Hyperparameters — LAYER 3 MODIFIES THIS SECTION
# ---------------------------------------------------------------------------

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = 256
DROPOUT = 0.2
OPTIMIZER = "adamw"  # adam, sgd, adamw
LR_SCHEDULE = "cosine"  # cosine, constant, step
WARMUP_STEPS = 50
LABEL_SMOOTHING = 0.0

# ---------------------------------------------------------------------------
# Model Architecture — LOCKED (simple MLP, do not modify)
# ---------------------------------------------------------------------------

HIDDEN_DIMS = [64, 32]
ACTIVATION = "gelu"


def get_activation() -> nn.Module:
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
    }
    return activations.get(ACTIVATION, nn.GELU())


class StockPredictor(nn.Module):
    """Simple MLP for binary stock direction prediction."""

    def __init__(self, input_dim: int):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in HIDDEN_DIMS:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_activation())
            layers.append(nn.Dropout(DROPOUT))
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train() -> dict:
    # Set seed for reproducibility when SEED > 0
    if SEED > 0:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load data
    ticker_dir = DATA_DIR / TICKER
    if not ticker_dir.exists():
        print(f"ERROR: Data not found at {ticker_dir}. Run prepare.py first.", flush=True)
        sys.exit(1)

    X_train = np.load(ticker_dir / "X_train.npy")
    y_train = np.load(ticker_dir / "y_train.npy")
    X_val = np.load(ticker_dir / "X_val.npy")
    y_val = np.load(ticker_dir / "y_val.npy")

    metadata = json.loads((ticker_dir / "metadata.json").read_text())
    n_features = X_train.shape[1]

    print(f"ticker:     {TICKER}", flush=True)
    print(f"device:     {device}", flush=True)
    print(f"n_features: {n_features}", flush=True)
    print(f"n_train:    {len(X_train)}", flush=True)
    print(f"n_val:      {len(X_val)}", flush=True)
    print(f"baseline:   {y_val.mean():.4f}", flush=True)

    # Normalize features (fit on train, apply to both)
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Model
    model = StockPredictor(n_features).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"n_params:   {n_params}", flush=True)

    # Optimizer
    if OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=0.9)
    elif OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Loss with label smoothing
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if LABEL_SMOOTHING > 0:
            targets = targets * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        return F.binary_cross_entropy_with_logits(logits, targets)

    # Training
    start_time = time.time()
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_state = None
    step = 0
    epoch = 0
    epochs_without_improvement = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed >= TIME_BUDGET:
            print(f"  Time budget reached ({TIME_BUDGET}s).", flush=True)
            break

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping: no val_loss improvement for {EARLY_STOP_PATIENCE} epochs.", flush=True)
            break

        model.train()
        epoch += 1

        for batch_X, batch_y in train_loader:
            elapsed = time.time() - start_time
            if elapsed >= TIME_BUDGET:
                break

            step += 1

            # Learning rate schedule
            if LR_SCHEDULE == "cosine":
                progress = min(elapsed / TIME_BUDGET, 1.0)
                if step <= WARMUP_STEPS:
                    lr = LEARNING_RATE * (step / max(WARMUP_STEPS, 1))
                else:
                    lr = LEARNING_RATE * 0.5 * (1 + np.cos(np.pi * progress))
            elif LR_SCHEDULE == "step":
                lr = LEARNING_RATE * (0.1 ** (elapsed / TIME_BUDGET))
            else:
                lr = LEARNING_RATE

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = compute_loss(logits, batch_y)

            if torch.isnan(loss) or loss.item() > 100:
                print("ERROR: Loss exploded, stopping early.", flush=True)
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Evaluate every epoch
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = F.binary_cross_entropy_with_logits(val_logits, y_val_t).item()
            val_probs = torch.sigmoid(val_logits)
            val_preds = (val_probs >= 0.5).float()
            val_acc = (val_preds == y_val_t).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 50 == 0:
            print(f"  epoch {epoch:4d} | step {step:6d} | val_acc {val_acc:.4f} | val_loss {val_loss:.4f} | best_loss {best_val_loss:.4f} | patience {epochs_without_improvement}/{EARLY_STOP_PATIENCE} | lr {lr:.2e}", flush=True)

    training_time = time.time() - start_time

    # Final evaluation with best model
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_preds = (val_probs >= 0.5).astype(np.float32)

        train_logits = model(X_train_t)
        train_probs = torch.sigmoid(train_logits).cpu().numpy()
        train_preds = (train_probs >= 0.5).astype(np.float32)

    # Metrics
    val_acc = float((val_preds == y_val).mean())
    train_acc = float((train_preds == y_train).mean())
    val_log_loss = float(-np.mean(
        y_val * np.log(np.clip(val_probs, 1e-7, 1 - 1e-7))
        + (1 - y_val) * np.log(np.clip(1 - val_probs, 1e-7, 1 - 1e-7))
    ))

    # Print results in grep-friendly format (like autoresearch)
    print(f"val_accuracy:       {val_acc:.6f}", flush=True)
    print(f"val_log_loss:       {val_log_loss:.6f}", flush=True)
    print(f"train_accuracy:     {train_acc:.6f}", flush=True)
    print(f"best_val_accuracy:  {best_val_acc:.6f}", flush=True)
    print(f"baseline_accuracy:  {y_val.mean():.6f}", flush=True)
    print(f"training_seconds:   {training_time:.1f}", flush=True)
    print(f"total_epochs:       {epoch}", flush=True)
    print(f"total_steps:        {step}", flush=True)
    print(f"n_params:           {n_params}", flush=True)

    return {
        "val_accuracy": val_acc,
        "val_log_loss": val_log_loss,
        "train_accuracy": train_acc,
        "best_val_accuracy": best_val_acc,
        "baseline_accuracy": float(y_val.mean()),
        "training_seconds": training_time,
        "total_epochs": epoch,
        "total_steps": step,
        "n_params": n_params,
    }


if __name__ == "__main__":
    train()
