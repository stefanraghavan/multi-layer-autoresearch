"""Training for daily stock direction prediction.

Supports two model types:
- XGBoost (gradient boosted trees) — default, best for tabular data
- MLP (simple neural network) — set MODEL_TYPE=mlp

Usage:
    python train.py                         # XGBoost (default)
    MODEL_TYPE=mlp python train.py          # MLP
    TICKER=_combined python train.py        # Train on combined dataset
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants (fixed evaluation protocol — do not modify)
# ---------------------------------------------------------------------------

TICKER = os.environ.get("TICKER", "_combined").lower()
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))
SEED = int(os.environ.get("SEED", "0"))  # 0 = no fixed seed
MODEL_TYPE = os.environ.get("MODEL_TYPE", "xgboost").lower()  # xgboost or mlp

# ---------------------------------------------------------------------------
# XGBoost hyperparameters
# ---------------------------------------------------------------------------

XGB_N_ESTIMATORS = int(os.environ.get("XGB_N_ESTIMATORS", "500"))
XGB_MAX_DEPTH = int(os.environ.get("XGB_MAX_DEPTH", "4"))
XGB_LEARNING_RATE = float(os.environ.get("XGB_LEARNING_RATE", "0.05"))
XGB_SUBSAMPLE = float(os.environ.get("XGB_SUBSAMPLE", "0.8"))
XGB_COLSAMPLE_BYTREE = float(os.environ.get("XGB_COLSAMPLE_BYTREE", "0.8"))
XGB_MIN_CHILD_WEIGHT = int(os.environ.get("XGB_MIN_CHILD_WEIGHT", "5"))
XGB_REG_ALPHA = float(os.environ.get("XGB_REG_ALPHA", "0.1"))
XGB_REG_LAMBDA = float(os.environ.get("XGB_REG_LAMBDA", "1.0"))
XGB_EARLY_STOPPING = int(os.environ.get("XGB_EARLY_STOPPING", "50"))

# ---------------------------------------------------------------------------
# MLP hyperparameters (fallback)
# ---------------------------------------------------------------------------

MLP_LEARNING_RATE = 1e-3
MLP_BATCH_SIZE = 256
MLP_DROPOUT = 0.2
MLP_HIDDEN_DIMS = [64, 32]
MLP_TIME_BUDGET = 60
MLP_EARLY_STOP_PATIENCE = 200


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train, X_val, y_val, seed):
    """Train XGBoost classifier."""
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        reg_alpha=XGB_REG_ALPHA,
        reg_lambda=XGB_REG_LAMBDA,
        random_state=seed if seed > 0 else None,
        eval_metric="logloss",
        early_stopping_rounds=XGB_EARLY_STOPPING,
        verbosity=0,
        use_label_encoder=False,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_probs = model.predict_proba(X_val)[:, 1]
    train_probs = model.predict_proba(X_train)[:, 1]

    return model, train_probs, val_probs, model.best_iteration


def train_mlp(X_train, y_train, X_val, y_val, seed):
    """Train simple MLP (fallback)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    if seed > 0:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Normalize
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - train_mean) / train_std
    X_val_n = (X_val - train_mean) / train_std

    X_train_t = torch.tensor(X_train_n, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val_n, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=MLP_BATCH_SIZE, shuffle=True, drop_last=True)

    # Build model
    n_features = X_train.shape[1]
    layers = []
    prev_dim = n_features
    for hidden_dim in MLP_HIDDEN_DIMS:
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(MLP_DROPOUT),
        ])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, 1))
    model = nn.Sequential(*layers).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_LEARNING_RATE)

    start_time = time.time()
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    epoch = 0
    while True:
        if time.time() - start_time >= MLP_TIME_BUDGET:
            break
        if epochs_without_improvement >= MLP_EARLY_STOP_PATIENCE:
            break

        model.train()
        epoch += 1
        for batch_X, batch_y in train_loader:
            if time.time() - start_time >= MLP_TIME_BUDGET:
                break
            optimizer.zero_grad()
            logits = model(batch_X).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).squeeze(-1)
            val_loss = F.binary_cross_entropy_with_logits(val_logits, y_val_t).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(X_val_t).squeeze(-1)).cpu().numpy()
        train_probs = torch.sigmoid(model(X_train_t).squeeze(-1)).cpu().numpy()

    return model, train_probs, val_probs, epoch


def train() -> dict:
    seed = SEED if SEED > 0 else 0

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
    print(f"model_type: {MODEL_TYPE}", flush=True)
    print(f"n_features: {n_features}", flush=True)
    print(f"n_train:    {len(X_train)}", flush=True)
    print(f"n_val:      {len(X_val)}", flush=True)
    print(f"baseline:   {y_val.mean():.4f}", flush=True)

    start_time = time.time()

    if MODEL_TYPE == "xgboost":
        model, train_probs, val_probs, n_iters = train_xgboost(X_train, y_train, X_val, y_val, seed)
        print(f"xgb_iters:  {n_iters}", flush=True)
    else:
        model, train_probs, val_probs, n_iters = train_mlp(X_train, y_train, X_val, y_val, seed)

    training_time = time.time() - start_time

    # Metrics
    val_preds = (val_probs >= 0.5).astype(np.float32)
    train_preds = (train_probs >= 0.5).astype(np.float32)

    val_acc = float((val_preds == y_val).mean())
    train_acc = float((train_preds == y_train).mean())
    val_log_loss = float(-np.mean(
        y_val * np.log(np.clip(val_probs, 1e-7, 1 - 1e-7))
        + (1 - y_val) * np.log(np.clip(1 - val_probs, 1e-7, 1 - 1e-7))
    ))

    # Print results in grep-friendly format
    print(f"val_accuracy:       {val_acc:.6f}", flush=True)
    print(f"val_log_loss:       {val_log_loss:.6f}", flush=True)
    print(f"train_accuracy:     {train_acc:.6f}", flush=True)
    print(f"best_val_accuracy:  {val_acc:.6f}", flush=True)
    print(f"baseline_accuracy:  {y_val.mean():.6f}", flush=True)
    print(f"training_seconds:   {training_time:.1f}", flush=True)
    print(f"total_epochs:       {n_iters}", flush=True)
    print(f"n_params:           {n_features}", flush=True)

    # Feature importance (XGBoost only)
    if MODEL_TYPE == "xgboost" and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = metadata.get("feature_names", [f"f{i}" for i in range(n_features)])
        sorted_idx = np.argsort(importances)[::-1]
        print(f"\nTop 10 features by importance:", flush=True)
        for i in sorted_idx[:10]:
            print(f"  {feature_names[i]:40s} {importances[i]:.4f}", flush=True)

    return {
        "val_accuracy": val_acc,
        "val_log_loss": val_log_loss,
        "train_accuracy": train_acc,
        "best_val_accuracy": val_acc,
        "baseline_accuracy": float(y_val.mean()),
        "training_seconds": training_time,
        "total_epochs": n_iters,
        "n_params": n_features,
    }


if __name__ == "__main__":
    train()
