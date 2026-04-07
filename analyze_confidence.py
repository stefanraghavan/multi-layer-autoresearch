"""Analyze model confidence thresholds and abstention strategies.

Trains the current best XGBoost model and evaluates hit rate at various
confidence thresholds. Shows the tradeoff between accuracy and coverage.

Usage:
    python analyze_confidence.py
    python analyze_confidence.py --seeds 5    # Average over 5 seeds
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))
TICKER = os.environ.get("TICKER", "_combined").lower()


def train_and_analyze(seed: int = 42) -> dict:
    from xgboost import XGBClassifier

    ticker_dir = DATA_DIR / TICKER
    X_train = np.load(ticker_dir / "X_train.npy")
    y_train = np.load(ticker_dir / "y_train.npy")
    X_val = np.load(ticker_dir / "X_val.npy")
    y_val = np.load(ticker_dir / "y_val.npy")
    metadata = json.loads((ticker_dir / "metadata.json").read_text())

    model = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, random_state=seed,
        eval_metric="logloss", early_stopping_rounds=50, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_probs = model.predict_proba(X_val)[:, 1]

    results = {}
    for thresh in [0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70]:
        high_conf = (val_probs >= thresh) | (val_probs <= (1 - thresh))
        n = int(high_conf.sum())
        if n == 0:
            continue
        preds = (val_probs[high_conf] >= 0.5).astype(float)
        actuals = y_val[high_conf]
        results[thresh] = {
            "hit_rate": float((preds == actuals).mean()),
            "coverage": float(high_conf.mean()),
            "n_trades": n,
        }

    return results, val_probs, y_val, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to average over")
    args = parser.parse_args()

    ticker_dir = DATA_DIR / TICKER
    if not ticker_dir.exists():
        print(f"ERROR: Data not found at {ticker_dir}. Run prepare.py first.", flush=True)
        sys.exit(1)

    X_val = np.load(ticker_dir / "X_val.npy")
    y_val = np.load(ticker_dir / "y_val.npy")
    X_train = np.load(ticker_dir / "X_train.npy")
    metadata = json.loads((ticker_dir / "metadata.json").read_text())

    print(f"Ticker: {TICKER}", flush=True)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}", flush=True)
    print(f"Features: {metadata['n_features']}", flush=True)
    print(f"Baseline (majority class): {max(y_val.mean(), 1 - y_val.mean()):.3f}", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    print()

    # Aggregate across seeds
    all_results = {}
    seeds = list(range(1, args.seeds + 1))

    for seed in seeds:
        results, _, _, _ = train_and_analyze(seed)
        for thresh, metrics in results.items():
            if thresh not in all_results:
                all_results[thresh] = {"hit_rates": [], "coverages": [], "n_trades": []}
            all_results[thresh]["hit_rates"].append(metrics["hit_rate"])
            all_results[thresh]["coverages"].append(metrics["coverage"])
            all_results[thresh]["n_trades"].append(metrics["n_trades"])

    print(f"{'Threshold':>12s} | {'Avg Hit Rate':>12s} | {'Std':>8s} | {'Coverage':>10s} | {'Avg Trades':>10s} | {'Min HR':>8s} | {'Max HR':>8s}")
    print("-" * 85)

    for thresh in sorted(all_results.keys()):
        m = all_results[thresh]
        hr = np.array(m["hit_rates"])
        cov = np.array(m["coverages"])
        nt = np.array(m["n_trades"])
        print(f"{thresh:>12.2f} | {hr.mean():>12.4f} | {hr.std():>8.4f} | {cov.mean():>10.1%} | {nt.mean():>10.0f} | {hr.min():>8.4f} | {hr.max():>8.4f}")

    # Feature importance from last model
    print()
    results, val_probs, y_val, metadata = train_and_analyze(42)
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        eval_metric="logloss", early_stopping_rounds=50, verbosity=0,
    )
    X_train = np.load(DATA_DIR / TICKER / "X_train.npy")
    y_train = np.load(DATA_DIR / TICKER / "y_train.npy")
    X_val = np.load(DATA_DIR / TICKER / "X_val.npy")
    y_val = np.load(DATA_DIR / TICKER / "y_val.npy")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    feature_names = metadata.get("feature_names", [f"f{i}" for i in range(metadata["n_features"])])
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("Top 15 features by importance:")
    for i in sorted_idx[:15]:
        print(f"  {feature_names[i]:40s} {importances[i]:.4f}")


if __name__ == "__main__":
    main()
