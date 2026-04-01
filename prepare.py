"""Data preparation for daily stock direction prediction.

Downloads daily OHLCV data, computes baseline features, creates walk-forward
train/validation splits, and saves as numpy arrays.

Multi-ticker: each ticker is processed independently and saved to its own
directory. train.py can load one or combine multiple.

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
TRAIN_YEARS = int(os.environ.get("TRAIN_YEARS", "7"))
VAL_YEARS = int(os.environ.get("VALIDATION_YEARS", "1"))

# Minimum number of trading days required
MIN_TRADING_DAYS = 252 * (TRAIN_YEARS + VAL_YEARS)

# Output directory (relative to script location or DATA_DIR)
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))

# Default ticker universe: broad market ETFs + sector ETFs + liquid large-caps
DEFAULT_TICKERS = (
    # Broad market
    "SPY,QQQ,IWM,DIA,"
    # Sector ETFs
    "XLF,XLE,XLK,XLV,XLI,XLP,XLU,XLB,XLC,XLRE,"
    # Liquid large-caps
    "AAPL,MSFT,AMZN,GOOGL,META,NVDA,TSLA,JPM,JNJ,V"
)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

# All features use data from day T-1 and earlier to predict day T's direction.
# This is the point-in-time rule: no same-day data leakage.

def compute_returns(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    """Compute log returns over multiple periods using PREVIOUS day's close."""
    features = pd.DataFrame(index=df.index)
    prev_close = df["Close"].shift(1)
    for p in periods:
        features[f"return_{p}d"] = np.log(prev_close / df["Close"].shift(p))
    return features


def compute_volatility(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute rolling volatility using PREVIOUS day's returns."""
    daily_ret = np.log(df["Close"].shift(1) / df["Close"].shift(2))
    features = pd.DataFrame(index=df.index)
    for w in windows:
        features[f"volatility_{w}d"] = daily_ret.rolling(w).std()
    return features


def compute_volume_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute volume-based features using PREVIOUS day's volume."""
    features = pd.DataFrame(index=df.index)
    prev_volume = df["Volume"].shift(1)
    for w in windows:
        features[f"volume_ratio_{w}d"] = prev_volume / df["Volume"].shift(1).rolling(w).mean()
    return features


def compute_price_position(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute price position within rolling high-low range using PREVIOUS day."""
    features = pd.DataFrame(index=df.index)
    prev_close = df["Close"].shift(1)
    for w in windows:
        rolling_high = df["High"].shift(1).rolling(w).max()
        rolling_low = df["Low"].shift(1).rolling(w).min()
        rng = rolling_high - rolling_low
        features[f"price_position_{w}d"] = np.where(
            rng > 0,
            (prev_close - rolling_low) / rng,
            0.5,
        )
    return features


def compute_moving_average_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute price relative to moving averages using PREVIOUS day's close."""
    features = pd.DataFrame(index=df.index)
    prev_close = df["Close"].shift(1)
    for w in windows:
        ma = df["Close"].shift(1).rolling(w).mean()
        features[f"price_vs_ma_{w}d"] = (prev_close - ma) / ma
    return features


def compute_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute overnight gap (previous day's open vs day-before close)."""
    features = pd.DataFrame(index=df.index)
    features["overnight_gap"] = np.log(df["Open"].shift(1) / df["Close"].shift(2))
    return features


def compute_intraday_range(df: pd.DataFrame) -> pd.DataFrame:
    """Compute intraday range as fraction of close (previous day)."""
    features = pd.DataFrame(index=df.index)
    features["intraday_range"] = (df["High"].shift(1) - df["Low"].shift(1)) / df["Close"].shift(1)
    return features


def compute_rsi(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute Relative Strength Index using PREVIOUS day's data."""
    daily_ret = df["Close"].shift(1) - df["Close"].shift(2)
    features = pd.DataFrame(index=df.index)
    for w in windows:
        gain = daily_ret.clip(lower=0).rolling(w).mean()
        loss = (-daily_ret.clip(upper=0)).rolling(w).mean()
        rs = gain / (loss + 1e-10)
        features[f"rsi_{w}d"] = 100 - (100 / (1 + rs))
    return features


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MACD and signal line using PREVIOUS day's close."""
    prev_close = df["Close"].shift(1)
    ema12 = prev_close.ewm(span=12, adjust=False).mean()
    ema26 = prev_close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    features = pd.DataFrame(index=df.index)
    features["macd"] = macd_line / prev_close  # normalize by price
    features["macd_signal"] = signal_line / prev_close
    features["macd_histogram"] = (macd_line - signal_line) / prev_close
    return features


def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute calendar-based features (day of week, month)."""
    features = pd.DataFrame(index=df.index)
    dates = pd.DatetimeIndex(df.index)
    # Encode as sin/cos to capture cyclical nature
    features["dow_sin"] = np.sin(2 * np.pi * dates.dayofweek / 5)
    features["dow_cos"] = np.cos(2 * np.pi * dates.dayofweek / 5)
    features["month_sin"] = np.sin(2 * np.pi * dates.month / 12)
    features["month_cos"] = np.cos(2 * np.pi * dates.month / 12)
    return features


def compute_streak_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute consecutive up/down day streaks using PREVIOUS days."""
    daily_up = (df["Close"].shift(1) > df["Close"].shift(2)).astype(float)
    features = pd.DataFrame(index=df.index)

    # Count consecutive up/down days (capped at 10)
    streak = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        if daily_up.iloc[i] == 1:
            streak.iloc[i] = max(streak.iloc[i-1], 0) + 1
        else:
            streak.iloc[i] = min(streak.iloc[i-1], 0) - 1
    features["streak"] = streak.clip(-10, 10) / 10  # normalize to [-1, 1]
    return features


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the full feature matrix from OHLCV data.

    *** LAYER 1 MODIFIES THIS FUNCTION ***
    The Feature Research agent adds/removes feature computations here.
    """
    parts = []

    # Returns at multiple horizons
    parts.append(compute_returns(df, periods=[2, 3, 5, 10, 20]))

    # Volatility at multiple windows
    parts.append(compute_volatility(df, windows=[5, 10, 20, 60]))

    # Volume features
    parts.append(compute_volume_features(df, windows=[5, 10, 20]))

    # Price position in range
    parts.append(compute_price_position(df, windows=[5, 10, 20, 60]))

    # Price vs moving averages
    parts.append(compute_moving_average_features(df, windows=[10, 20, 50, 200]))

    # Gap and range
    parts.append(compute_gap(df))
    parts.append(compute_intraday_range(df))

    # RSI
    parts.append(compute_rsi(df, windows=[7, 14]))

    # MACD
    parts.append(compute_macd(df))

    # Calendar features
    parts.append(compute_calendar_features(df))

    # Streak features
    parts.append(compute_streak_features(df))

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
    """Compute evaluation metrics for binary classification."""
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
        "baseline_accuracy": float(y_true.mean()),
    }


# ---------------------------------------------------------------------------
# Main: prepare and save dataset
# ---------------------------------------------------------------------------

def main() -> None:
    tickers_str = os.environ.get("TICKERS", DEFAULT_TICKERS)
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

    total_years = TRAIN_YEARS + VAL_YEARS + 1  # extra year for feature warmup
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * total_years)).strftime("%Y-%m-%d")

    print(f"Preparing data for {len(tickers)} tickers: {tickers}", flush=True)
    print(f"Date range: {start_date} to {end_date}", flush=True)
    print(f"Train: {TRAIN_YEARS} years, Validation: {VAL_YEARS} year(s)", flush=True)

    # Download
    raw_data = download_data(tickers, start_date, end_date)
    if not raw_data:
        print("ERROR: No data downloaded. Check tickers and internet connection.", flush=True)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Process each ticker and also build a combined dataset
    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []
    combined_feature_names = None

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

        if len(X) < 252 * 2:
            print(f"  WARNING: Only {len(X)} samples for {ticker}, skipping (need at least {252*2})", flush=True)
            continue

        print(f"  Features: {len(feature_names)}", flush=True)
        print(f"  Samples: {len(X)}", flush=True)
        print(f"  Up-day fraction: {y.mean():.3f}", flush=True)

        # Create walk-forward split
        split = create_walk_forward_split(X, y, dates)

        print(f"  Train: {len(split['X_train'])} days", flush=True)
        print(f"  Val: {len(split['X_val'])} days", flush=True)

        # Save per-ticker
        ticker_dir = DATA_DIR / ticker.lower()
        ticker_dir.mkdir(parents=True, exist_ok=True)

        np.save(ticker_dir / "X_train.npy", split["X_train"])
        np.save(ticker_dir / "y_train.npy", split["y_train"])
        np.save(ticker_dir / "X_val.npy", split["X_val"])
        np.save(ticker_dir / "y_val.npy", split["y_val"])

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

        # Accumulate for combined dataset
        all_X_train.append(split["X_train"])
        all_y_train.append(split["y_train"])
        all_X_val.append(split["X_val"])
        all_y_val.append(split["y_val"])
        combined_feature_names = feature_names

    # Save combined dataset (all tickers stacked)
    if all_X_train:
        combined_dir = DATA_DIR / "_combined"
        combined_dir.mkdir(parents=True, exist_ok=True)

        X_train_all = np.concatenate(all_X_train, axis=0)
        y_train_all = np.concatenate(all_y_train, axis=0)
        X_val_all = np.concatenate(all_X_val, axis=0)
        y_val_all = np.concatenate(all_y_val, axis=0)

        np.save(combined_dir / "X_train.npy", X_train_all)
        np.save(combined_dir / "y_train.npy", y_train_all)
        np.save(combined_dir / "X_val.npy", X_val_all)
        np.save(combined_dir / "y_val.npy", y_val_all)

        metadata = {
            "ticker": "_combined",
            "tickers_included": [t for t in raw_data.keys()],
            "n_tickers": len(all_X_train),
            "feature_names": combined_feature_names,
            "n_features": len(combined_feature_names),
            "n_train": int(len(X_train_all)),
            "n_val": int(len(X_val_all)),
            "up_day_fraction_train": float(y_train_all.mean()),
            "up_day_fraction_val": float(y_val_all.mean()),
        }
        (combined_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        print(f"\nCombined dataset: {len(X_train_all)} train, {len(X_val_all)} val samples across {len(all_X_train)} tickers", flush=True)

    print("\nData preparation complete.", flush=True)


if __name__ == "__main__":
    main()
