"""Data preparation for daily stock direction prediction.

Downloads daily OHLCV data, computes baseline features, creates walk-forward
train/validation splits, and saves as numpy arrays.

This script is modified by the Layer 1 (Feature Research) agent to add/remove
features. The feature configuration is stored in features.json.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (DO NOT MODIFY — fixed evaluation protocol)
# ---------------------------------------------------------------------------

# Walk-forward validation: train on TRAIN_YEARS, validate on VAL_YEARS
TRAIN_YEARS = int(os.environ.get("TRAIN_YEARS", "3"))
VAL_YEARS = int(os.environ.get("VALIDATION_YEARS", "1"))

# Minimum number of trading days required
MIN_TRADING_DAYS = 252 * (TRAIN_YEARS + VAL_YEARS)

# Output directory (relative to script location or DATA_DIR)
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

# FEATURES_CONFIG: This section defines which features to compute.
# The Layer 1 (Feature Research) agent modifies this section.
# Each feature is a dict with 'name', 'compute' function reference, and 'params'.

def compute_returns(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    """Compute log returns over multiple periods."""
    features = pd.DataFrame(index=df.index)
    for p in periods:
        features[f"return_{p}d"] = np.log(df["Close"] / df["Close"].shift(p))
    return features


def compute_volatility(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute rolling volatility (std of daily returns)."""
    daily_ret = np.log(df["Close"] / df["Close"].shift(1))
    features = pd.DataFrame(index=df.index)
    for w in windows:
        features[f"volatility_{w}d"] = daily_ret.rolling(w).std()
    return features


def compute_volume_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute volume-based features."""
    features = pd.DataFrame(index=df.index)
    for w in windows:
        features[f"volume_ratio_{w}d"] = df["Volume"] / df["Volume"].rolling(w).mean()
    return features


def compute_price_position(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute price position within rolling high-low range."""
    features = pd.DataFrame(index=df.index)
    for w in windows:
        rolling_high = df["High"].rolling(w).max()
        rolling_low = df["Low"].rolling(w).min()
        rng = rolling_high - rolling_low
        features[f"price_position_{w}d"] = np.where(
            rng > 0,
            (df["Close"] - rolling_low) / rng,
            0.5,
        )
    return features


def compute_moving_average_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute price relative to moving averages."""
    features = pd.DataFrame(index=df.index)
    for w in windows:
        ma = df["Close"].rolling(w).mean()
        features[f"price_vs_ma_{w}d"] = (df["Close"] - ma) / ma
    return features


def compute_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute overnight gap (open vs previous close)."""
    features = pd.DataFrame(index=df.index)
    features["overnight_gap"] = np.log(df["Open"] / df["Close"].shift(1))
    return features


def compute_intraday_range(df: pd.DataFrame) -> pd.DataFrame:
    """Compute intraday range as fraction of close."""
    features = pd.DataFrame(index=df.index)
    features["intraday_range"] = (df["High"] - df["Low"]) / df["Close"]
    return features


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the full feature matrix from OHLCV data.

    *** LAYER 1 MODIFIES THIS FUNCTION ***
    The Feature Research agent adds/removes feature computations here.
    """
    parts = []

    # Returns at multiple horizons
    parts.append(compute_returns(df, periods=[1, 2, 3, 5, 10, 20]))

    # Volatility at multiple windows
    parts.append(compute_volatility(df, windows=[5, 10, 20, 60]))

    # Volume features
    parts.append(compute_volume_features(df, windows=[5, 10, 20]))

    # Price position in range
    parts.append(compute_price_position(df, windows=[5, 10, 20, 60]))

    # Price vs moving averages
    parts.append(compute_moving_average_features(df, windows=[5, 10, 20, 50, 200]))

    # Gap and range
    parts.append(compute_gap(df))
    parts.append(compute_intraday_range(df))

    features = pd.concat(parts, axis=1)
    return features


# ---------------------------------------------------------------------------
# Target computation
# ---------------------------------------------------------------------------

def compute_target(df: pd.DataFrame) -> pd.Series:
    """Binary target: 1 if close > previous close, 0 otherwise."""
    return (df["Close"] > df["Close"].shift(1)).astype(np.float32)


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_data(tickers: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV data for given tickers."""
    import yfinance as yf

    data = {}
    for ticker in tickers:
        print(f"Downloading {ticker}...", flush=True)
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            print(f"  WARNING: No data for {ticker}", flush=True)
            continue
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[ticker] = df
        print(f"  {len(df)} trading days", flush=True)

    return data


# ---------------------------------------------------------------------------
# Walk-forward split
# ---------------------------------------------------------------------------

def create_walk_forward_split(
    features: np.ndarray,
    targets: np.ndarray,
    dates: np.ndarray,
    train_years: int = TRAIN_YEARS,
    val_years: int = VAL_YEARS,
) -> dict:
    """Create walk-forward train/validation split.

    Uses the most recent val_years as validation, everything before as training.
    """
    n = len(dates)
    val_days = 252 * val_years
    train_end = n - val_days

    if train_end < 252:
        raise ValueError(f"Insufficient data: {n} days, need at least {252 + val_days}")

    return {
        "X_train": features[:train_end],
        "y_train": targets[:train_end],
        "dates_train": dates[:train_end],
        "X_val": features[train_end:],
        "y_val": targets[train_end:],
        "dates_val": dates[train_end:],
    }


# ---------------------------------------------------------------------------
# Evaluation helpers (used by train.py)
# ---------------------------------------------------------------------------

def evaluate_predictions(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Compute evaluation metrics for binary classification.

    Returns dict with accuracy, log_loss, brier_score, profit_weighted_accuracy.
    """
    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

    y_pred = (y_pred_proba >= 0.5).astype(np.float32)
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, np.clip(y_pred_proba, 1e-7, 1 - 1e-7))
    brier = brier_score_loss(y_true, y_pred_proba)

    return {
        "accuracy": float(acc),
        "log_loss": float(ll),
        "brier_score": float(brier),
        "n_samples": int(len(y_true)),
        "baseline_accuracy": float(y_true.mean()),  # fraction of up days
    }


# ---------------------------------------------------------------------------
# Main: prepare and save dataset
# ---------------------------------------------------------------------------

def main() -> None:
    tickers_str = os.environ.get("TICKERS", "SPY")
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

    total_years = TRAIN_YEARS + VAL_YEARS + 1  # extra year for feature warmup
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * total_years)).strftime("%Y-%m-%d")

    print(f"Preparing data for: {tickers}", flush=True)
    print(f"Date range: {start_date} to {end_date}", flush=True)
    print(f"Train: {TRAIN_YEARS} years, Validation: {VAL_YEARS} year(s)", flush=True)

    # Download
    raw_data = download_data(tickers, start_date, end_date)
    if not raw_data:
        print("ERROR: No data downloaded. Check tickers and internet connection.", flush=True)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for ticker, df in raw_data.items():
        print(f"\nProcessing {ticker}...", flush=True)

        # Compute features and target
        features_df = compute_all_features(df)
        target = compute_target(df)

        # Align and drop NaN rows (features have warmup periods)
        combined = pd.concat([features_df, target.rename("target")], axis=1).dropna()

        feature_names = [c for c in combined.columns if c != "target"]
        X = combined[feature_names].values.astype(np.float32)
        y = combined["target"].values.astype(np.float32)
        dates = combined.index.values

        print(f"  Features: {len(feature_names)}", flush=True)
        print(f"  Samples: {len(X)}", flush=True)
        print(f"  Up-day fraction: {y.mean():.3f}", flush=True)

        # Create walk-forward split
        split = create_walk_forward_split(X, y, dates)

        print(f"  Train: {len(split['X_train'])} days", flush=True)
        print(f"  Val: {len(split['X_val'])} days", flush=True)

        # Save
        ticker_dir = DATA_DIR / ticker.lower()
        ticker_dir.mkdir(parents=True, exist_ok=True)

        np.save(ticker_dir / "X_train.npy", split["X_train"])
        np.save(ticker_dir / "y_train.npy", split["y_train"])
        np.save(ticker_dir / "X_val.npy", split["X_val"])
        np.save(ticker_dir / "y_val.npy", split["y_val"])

        # Save metadata
        metadata = {
            "ticker": ticker,
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "n_train": int(len(split["X_train"])),
            "n_val": int(len(split["X_val"])),
            "train_date_range": [str(split["dates_train"][0])[:10], str(split["dates_train"][-1])[:10]],
            "val_date_range": [str(split["dates_val"][0])[:10], str(split["dates_val"][-1])[:10]],
            "up_day_fraction_train": float(split["y_train"].mean()),
            "up_day_fraction_val": float(split["y_val"].mean()),
        }
        (ticker_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"  Saved to {ticker_dir}/", flush=True)

    print("\nData preparation complete.", flush=True)


if __name__ == "__main__":
    main()
