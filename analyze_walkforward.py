"""Rolling walk-forward validation with confidence thresholds.

Trains on expanding windows, validates on the next year, rolls forward.
Tests whether the 70-75% high-conviction hit rate holds across regimes.

Usage:
    python analyze_walkforward.py
    python analyze_walkforward.py --min-train-years 4 --val-years 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))
TICKER = os.environ.get("TICKER", "_combined").lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-train-years", type=int, default=4, help="Minimum training years before first validation")
    parser.add_argument("--val-years", type=int, default=1, help="Validation window size in years")
    parser.add_argument("--step-years", type=int, default=1, help="How far to roll forward each step")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per window")
    args = parser.parse_args()

    ticker_dir = DATA_DIR / TICKER
    if not ticker_dir.exists():
        print(f"ERROR: Data not found at {ticker_dir}. Run prepare.py first.", flush=True)
        sys.exit(1)

    # Load all data (train + val combined for re-splitting)
    X_train = np.load(ticker_dir / "X_train.npy")
    y_train = np.load(ticker_dir / "y_train.npy")
    X_val = np.load(ticker_dir / "X_val.npy")
    y_val = np.load(ticker_dir / "y_val.npy")
    metadata = json.loads((ticker_dir / "metadata.json").read_text())

    # Combine all data
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    n_total = len(X_all)

    # We need date info to split by year. Estimate from metadata.
    # Each ticker contributes ~252 event-window days per year
    # Combined dataset has 30 tickers, so ~30 * ~40 event days/year = ~1200/year
    train_range = metadata.get("train_date_range", ["2016-01-01", "2024-01-01"])
    val_range = metadata.get("val_date_range", ["2024-01-01", "2026-01-01"])

    # Approximate: distribute samples evenly across the date range
    n_train = len(X_train)
    n_val = len(X_val)
    total_years = 10  # approximate total span
    samples_per_year = n_total / total_years

    print(f"Total samples: {n_total} (train={n_train}, val={n_val})", flush=True)
    print(f"Approx samples/year: {samples_per_year:.0f}", flush=True)
    print(f"Min train years: {args.min_train_years}, Val window: {args.val_years}yr, Step: {args.step_years}yr", flush=True)
    print(f"Seeds per window: {args.seeds}", flush=True)
    print()

    from xgboost import XGBClassifier

    # Rolling walk-forward
    windows = []
    train_start = 0
    val_size = int(samples_per_year * args.val_years)
    min_train_size = int(samples_per_year * args.min_train_years)
    step_size = int(samples_per_year * args.step_years)

    train_end = min_train_size
    while train_end + val_size <= n_total:
        val_start = train_end
        val_end = min(train_end + val_size, n_total)
        windows.append((train_start, train_end, val_start, val_end))
        train_end += step_size

    print(f"Number of walk-forward windows: {len(windows)}", flush=True)
    print()

    # Results per window and threshold
    thresholds = [0.50, 0.54, 0.55, 0.56, 0.58, 0.60]
    all_window_results = []

    for w_idx, (tr_start, tr_end, v_start, v_end) in enumerate(windows):
        X_tr = X_all[tr_start:tr_end]
        y_tr = y_all[tr_start:tr_end]
        X_v = X_all[v_start:v_end]
        y_v = y_all[v_start:v_end]

        approx_year = 2016 + args.min_train_years + w_idx * args.step_years
        print(f"Window {w_idx+1}/{len(windows)}: train={tr_end-tr_start}, val={v_end-v_start} (approx val year ~{approx_year})", flush=True)

        window_results = {}
        for thresh in thresholds:
            hit_rates = []
            n_trades_list = []

            for seed in range(1, args.seeds + 1):
                model = XGBClassifier(
                    n_estimators=500, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                    reg_alpha=0.1, reg_lambda=1.0, random_state=seed,
                    eval_metric="logloss", early_stopping_rounds=50, verbosity=0,
                )
                model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
                probs = model.predict_proba(X_v)[:, 1]

                high_conf = (probs >= thresh) | (probs <= (1 - thresh))
                n = int(high_conf.sum())
                if n > 0:
                    preds = (probs[high_conf] >= 0.5).astype(float)
                    hr = float((preds == y_v[high_conf]).mean())
                else:
                    hr = 0.5
                    n = 0
                hit_rates.append(hr)
                n_trades_list.append(n)

            window_results[thresh] = {
                "avg_hit_rate": float(np.mean(hit_rates)),
                "std_hit_rate": float(np.std(hit_rates)),
                "avg_n_trades": float(np.mean(n_trades_list)),
            }

        all_window_results.append({
            "window": w_idx + 1,
            "approx_val_year": approx_year,
            "train_size": tr_end - tr_start,
            "val_size": v_end - v_start,
            "results": window_results,
        })

        # Print summary for this window
        for thresh in [0.50, 0.55, 0.56]:
            r = window_results[thresh]
            print(f"  thresh={thresh:.2f}: hit_rate={r['avg_hit_rate']:.3f} ± {r['std_hit_rate']:.3f}, trades={r['avg_n_trades']:.0f}", flush=True)
        print()

    # Aggregate across all windows
    print("=" * 80)
    print("AGGREGATE ACROSS ALL WINDOWS")
    print("=" * 80)
    print()
    print(f"{'Threshold':>12s} | {'Avg HR':>8s} | {'Min HR':>8s} | {'Max HR':>8s} | {'Std':>8s} | {'Avg Trades':>10s} | {'Windows':>8s}")
    print("-" * 75)

    for thresh in thresholds:
        hrs = [w["results"][thresh]["avg_hit_rate"] for w in all_window_results if thresh in w["results"]]
        trades = [w["results"][thresh]["avg_n_trades"] for w in all_window_results if thresh in w["results"]]
        if hrs:
            print(f"{thresh:>12.2f} | {np.mean(hrs):>8.3f} | {np.min(hrs):>8.3f} | {np.max(hrs):>8.3f} | {np.std(hrs):>8.3f} | {np.mean(trades):>10.0f} | {len(hrs):>8d}")

    # Per-window detail for key thresholds
    print()
    print("PER-WINDOW DETAIL (threshold=0.55)")
    print("-" * 60)
    for w in all_window_results:
        r = w["results"].get(0.55, {})
        print(f"  Window {w['window']} (~{w['approx_val_year']}): HR={r.get('avg_hit_rate', 0):.3f}, trades={r.get('avg_n_trades', 0):.0f}")

    print()
    print("PER-WINDOW DETAIL (threshold=0.56)")
    print("-" * 60)
    for w in all_window_results:
        r = w["results"].get(0.56, {})
        print(f"  Window {w['window']} (~{w['approx_val_year']}): HR={r.get('avg_hit_rate', 0):.3f}, trades={r.get('avg_n_trades', 0):.0f}")


if __name__ == "__main__":
    main()
