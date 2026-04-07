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
TRAIN_YEARS = int(os.environ.get("TRAIN_YEARS", "8"))
VAL_YEARS = int(os.environ.get("VALIDATION_YEARS", "2"))

# Minimum number of trading days required
MIN_TRADING_DAYS = 252 * (TRAIN_YEARS + VAL_YEARS)

# Output directory (relative to script location or DATA_DIR)
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))

# Event window: only include days within EVENT_WINDOW_DAYS of an earnings announcement
EVENT_WINDOW_DAYS = int(os.environ.get("EVENT_WINDOW_DAYS", "10"))

# Default ticker universe: ~100 liquid S&P 500 stocks with FactSet coverage
DEFAULT_TICKERS = (
    # Tech (20)
    "AAPL,MSFT,AMZN,GOOGL,META,NVDA,TSLA,CRM,ADBE,ORCL,"
    "INTC,AMD,CSCO,AVGO,QCOM,TXN,NOW,INTU,AMAT,MU,"
    # Financials (15)
    "JPM,BAC,GS,MS,V,MA,WFC,C,AXP,BLK,"
    "SCHW,USB,PNC,TFC,COF,"
    # Healthcare (15)
    "JNJ,UNH,PFE,ABT,TMO,ABBV,MRK,LLY,MDT,DHR,"
    "BMY,AMGN,GILD,ISRG,SYK,"
    # Consumer Discretionary (10)
    "HD,MCD,NKE,SBUX,TJX,LOW,TGT,ROST,MAR,YUM,"
    # Consumer Staples (8)
    "WMT,PG,KO,PEP,COST,CL,MO,GIS,"
    # Industrials (10)
    "CAT,BA,UPS,HON,UNP,RTX,DE,GE,LMT,MMM,"
    # Energy (6)
    "XOM,CVX,COP,SLB,EOG,MPC,"
    # Utilities (4)
    "NEE,DUK,SO,D,"
    # Real Estate (4)
    "AMT,PLD,CCI,EQIX,"
    # Materials (4)
    "LIN,APD,SHW,ECL,"
    # Communication (4)
    "DIS,CMCSA,NFLX,T"
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


def compute_volatility_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Compute short-vs-long volatility regime using PREVIOUS day's returns."""
    daily_ret = np.log(df["Close"].shift(1) / df["Close"].shift(2))
    features = pd.DataFrame(index=df.index)
    vol_5 = daily_ret.rolling(5).std()
    vol_10 = daily_ret.rolling(10).std()
    vol_60 = daily_ret.rolling(60).std()
    features["volatility_ratio_5_60"] = vol_5 / (vol_60 + 1e-10)
    features["volatility_ratio_10_60"] = vol_10 / (vol_60 + 1e-10)
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


def compute_factset_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Load cached FactSet data and compute fundamental features.

    Features computed:
    - estimate_revision_30d: % change in consensus estimate over 30 days
    - estimate_revision_90d: % change in consensus estimate over 90 days
    - analyst_count: number of analysts covering
    - last_surprise_pct: most recent earnings surprise (actual vs estimate)
    - avg_surprise_pct: average surprise over last 4 quarters

    All values are forward-filled to daily and shifted by 1 day for point-in-time.
    Tickers without cached data get all zeros (neutral).
    """
    features = pd.DataFrame(index=df.index)
    factset_cols = [
        "estimate_revision_30d", "estimate_revision_90d",
        "analyst_count", "last_surprise_pct", "avg_surprise_pct",
    ]

    # Try to load cached FactSet data
    cache_path = DATA_DIR / ticker.lower() / "factset_cache.json"
    if not cache_path.exists():
        for col in factset_cols:
            features[col] = 0.0
        return features

    try:
        cache = json.loads(cache_path.read_text())
    except Exception:
        for col in factset_cols:
            features[col] = 0.0
        return features

    if cache.get("is_etf", False):
        for col in factset_cols:
            features[col] = 0.0
        return features

    # Consensus estimate revision momentum
    consensus = cache.get("consensus_time_series", [])
    if consensus:
        est_df = pd.DataFrame(consensus)
        est_df["date"] = pd.to_datetime(est_df["date"])
        est_df = est_df.set_index("date").sort_index()

        if "estimate" in est_df.columns and len(est_df) > 1:
            # Reindex to daily and forward-fill
            daily_est = est_df["estimate"].reindex(df.index, method="ffill")
            # Revision = % change over 30/90 day windows
            features["estimate_revision_30d"] = daily_est.pct_change(periods=30).shift(1).fillna(0)
            features["estimate_revision_90d"] = daily_est.pct_change(periods=90).shift(1).fillna(0)
        else:
            features["estimate_revision_30d"] = 0.0
            features["estimate_revision_90d"] = 0.0

        if "num_analysts" in est_df.columns:
            daily_analysts = est_df["num_analysts"].reindex(df.index, method="ffill")
            features["analyst_count"] = daily_analysts.shift(1).fillna(0) / 20.0  # normalize
        else:
            features["analyst_count"] = 0.0
    else:
        features["estimate_revision_30d"] = 0.0
        features["estimate_revision_90d"] = 0.0
        features["analyst_count"] = 0.0

    # Earnings surprise features
    actuals = cache.get("actuals", [])
    if actuals:
        surprises = [a["surprise_pct"] for a in actuals if a.get("surprise_pct") is not None]
        report_dates = [a.get("report_date") for a in actuals if a.get("report_date") and a.get("surprise_pct") is not None]

        if surprises and report_dates:
            # Build a time series of surprises at their report dates
            surprise_series = pd.Series(dtype=float, index=df.index)
            avg_surprise_series = pd.Series(dtype=float, index=df.index)

            sorted_actuals = sorted(
                [(rd, sp) for rd, sp in zip(report_dates, surprises) if rd],
                key=lambda x: x[0],
            )

            for i, (rd, sp) in enumerate(sorted_actuals):
                try:
                    rd_date = pd.Timestamp(rd)
                    if rd_date in surprise_series.index or rd_date <= surprise_series.index[-1]:
                        # Find the nearest date >= report_date
                        mask = surprise_series.index >= rd_date
                        if mask.any():
                            idx = surprise_series.index[mask][0]
                            surprise_series.loc[idx] = sp / 100.0  # normalize to fraction
                            # Rolling average of last 4 surprises
                            recent = [x[1] for x in sorted_actuals[max(0, i-3):i+1]]
                            avg_surprise_series.loc[idx] = sum(recent) / len(recent) / 100.0
                except Exception:
                    pass

            features["last_surprise_pct"] = surprise_series.ffill().shift(1).fillna(0)
            features["avg_surprise_pct"] = avg_surprise_series.ffill().shift(1).fillna(0)
        else:
            features["last_surprise_pct"] = 0.0
            features["avg_surprise_pct"] = 0.0
    else:
        features["last_surprise_pct"] = 0.0
        features["avg_surprise_pct"] = 0.0

    return features


def compute_all_features(df: pd.DataFrame, ticker: str = "SPY") -> pd.DataFrame:
    """Compute the full feature matrix from OHLCV + FactSet data.

    *** LAYER 1 MODIFIES THIS FUNCTION ***
    The Feature Research agent adds/removes feature computations here.
    """
    parts = []

    # Returns at multiple horizons
    parts.append(compute_returns(df, periods=[2, 3, 5, 10, 20]))

    # Volatility at multiple windows
    parts.append(compute_volatility(df, windows=[5, 10, 20, 60]))

    # Volatility regime ratios
    parts.append(compute_volatility_regime(df))

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

    # FactSet fundamental features (estimate revisions, surprise, short interest)
    parts.append(compute_factset_features(df, ticker))

    features = pd.concat(parts, axis=1)
    return features


# ---------------------------------------------------------------------------
# Earnings date extraction and event window filtering
# ---------------------------------------------------------------------------


def get_earnings_dates(ticker: str) -> list[pd.Timestamp]:
    """Extract earnings announcement dates from FactSet fundamentals cache.

    Uses eps_report_date from the quarterly fundamentals data.
    """
    cache_path = DATA_DIR / ticker.lower() / "factset_fundamentals_qtr.json"
    if not cache_path.exists():
        return []

    try:
        data = json.loads(cache_path.read_text())
    except Exception:
        return []

    records = data.get("records", [])
    dates = set()
    for r in records:
        rd = r.get("eps_report_date") or r.get("report_date")
        if rd:
            try:
                dates.add(pd.Timestamp(str(rd)[:10]))
            except Exception:
                pass

    return sorted(dates)


def filter_event_window(
    df: pd.DataFrame,
    earnings_dates: list[pd.Timestamp],
    window_days: int = EVENT_WINDOW_DAYS,
) -> pd.DataFrame:
    """Filter DataFrame to only include rows within window_days of an earnings date.

    Keeps rows from the earnings date through window_days trading days after.
    This captures the post-earnings reaction period where transcript/surprise
    features are most informative.
    """
    if not earnings_dates:
        return df  # no earnings dates known, keep everything

    mask = pd.Series(False, index=df.index)
    for ed in earnings_dates:
        # Include earnings day through window_days calendar days after
        window_start = ed
        window_end = ed + pd.Timedelta(days=window_days)
        mask |= (df.index >= window_start) & (df.index <= window_end)

    return df[mask]


# ---------------------------------------------------------------------------
# Sector mapping and target computation
# ---------------------------------------------------------------------------

SECTOR_MAP = {
    # Tech
    "AAPL": "tech", "MSFT": "tech", "AMZN": "tech", "GOOGL": "tech",
    "META": "tech", "NVDA": "tech", "TSLA": "tech", "CRM": "tech",
    "ADBE": "tech", "ORCL": "tech", "INTC": "tech", "AMD": "tech",
    "CSCO": "tech", "AVGO": "tech", "QCOM": "tech", "TXN": "tech",
    "NOW": "tech", "INTU": "tech", "AMAT": "tech", "MU": "tech",
    # Financials
    "JPM": "financials", "BAC": "financials", "GS": "financials",
    "MS": "financials", "V": "financials", "MA": "financials",
    "WFC": "financials", "C": "financials", "AXP": "financials",
    "BLK": "financials", "SCHW": "financials", "USB": "financials",
    "PNC": "financials", "TFC": "financials", "COF": "financials",
    # Healthcare
    "JNJ": "healthcare", "UNH": "healthcare", "PFE": "healthcare",
    "ABT": "healthcare", "TMO": "healthcare", "ABBV": "healthcare",
    "MRK": "healthcare", "LLY": "healthcare", "MDT": "healthcare",
    "DHR": "healthcare", "BMY": "healthcare", "AMGN": "healthcare",
    "GILD": "healthcare", "ISRG": "healthcare", "SYK": "healthcare",
    # Consumer Discretionary
    "HD": "consumer_disc", "MCD": "consumer_disc", "NKE": "consumer_disc",
    "SBUX": "consumer_disc", "TJX": "consumer_disc", "LOW": "consumer_disc",
    "TGT": "consumer_disc", "ROST": "consumer_disc", "MAR": "consumer_disc",
    "YUM": "consumer_disc",
    # Consumer Staples
    "WMT": "consumer_staples", "PG": "consumer_staples", "KO": "consumer_staples",
    "PEP": "consumer_staples", "COST": "consumer_staples", "CL": "consumer_staples",
    "MO": "consumer_staples", "GIS": "consumer_staples",
    # Industrials
    "CAT": "industrial", "BA": "industrial", "UPS": "industrial",
    "HON": "industrial", "UNP": "industrial", "RTX": "industrial",
    "DE": "industrial", "GE": "industrial", "LMT": "industrial", "MMM": "industrial",
    # Energy
    "XOM": "energy", "CVX": "energy", "COP": "energy",
    "SLB": "energy", "EOG": "energy", "MPC": "energy",
    # Utilities
    "NEE": "utilities", "DUK": "utilities", "SO": "utilities", "D": "utilities",
    # Real Estate
    "AMT": "real_estate", "PLD": "real_estate", "CCI": "real_estate", "EQIX": "real_estate",
    # Materials
    "LIN": "materials", "APD": "materials", "SHW": "materials", "ECL": "materials",
    # Communication
    "DIS": "communication", "CMCSA": "communication", "NFLX": "communication", "T": "communication",
}


def compute_sector_relative_target(
    all_data: dict[str, pd.DataFrame],
) -> dict[str, pd.Series]:
    """Compute sector-relative binary target for each ticker.

    Target = 1 if the stock's daily return exceeds its sector's average return, 0 otherwise.
    This strips out market/sector beta and asks: did this stock outperform its peers today?

    Uses previous-day-to-today close return for both the stock and the sector average.
    """
    # Compute daily returns for all tickers
    returns = {}
    for ticker, df in all_data.items():
        returns[ticker] = df["Close"].pct_change()

    # Group tickers by sector
    sectors: dict[str, list[str]] = {}
    for ticker in all_data:
        sector = SECTOR_MAP.get(ticker.upper(), "other")
        sectors.setdefault(sector, []).append(ticker)

    # Compute sector average return per day
    sector_avg_returns: dict[str, pd.Series] = {}
    for sector, tickers in sectors.items():
        sector_returns = pd.DataFrame({t: returns[t] for t in tickers if t in returns})
        sector_avg_returns[sector] = sector_returns.mean(axis=1)

    # Target: 1 if stock return > sector average return
    targets = {}
    for ticker in all_data:
        sector = SECTOR_MAP.get(ticker.upper(), "other")
        stock_ret = returns[ticker]
        sector_ret = sector_avg_returns[sector]
        # Align indices
        aligned = pd.DataFrame({"stock": stock_ret, "sector": sector_ret}).dropna()
        targets[ticker] = (aligned["stock"] > aligned["sector"]).astype(np.float32)
        targets[ticker].index = aligned.index

    return targets


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
    """Create walk-forward train/validation split by date.

    Uses the most recent val_years as validation, everything before as training.
    Splits by calendar date (not by count) to work with event-filtered data.
    """
    dates_ts = pd.DatetimeIndex(dates)
    cutoff_date = dates_ts.max() - pd.DateOffset(years=val_years)

    train_mask = dates_ts <= cutoff_date
    val_mask = dates_ts > cutoff_date

    if train_mask.sum() < 20:
        raise ValueError(f"Insufficient training data: {train_mask.sum()} samples before cutoff {cutoff_date.date()}")

    return {
        "X_train": features[train_mask],
        "y_train": targets[train_mask],
        "dates_train": dates[train_mask],
        "X_val": features[val_mask],
        "y_val": targets[val_mask],
        "dates_val": dates[val_mask],
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
    print(f"Event window: {EVENT_WINDOW_DAYS} days after earnings", flush=True)

    # Download
    raw_data = download_data(tickers, start_date, end_date)
    if not raw_data:
        print("ERROR: No data downloaded. Check tickers and internet connection.", flush=True)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Compute sector-relative targets (needs all tickers at once)
    print(f"\nComputing sector-relative targets...", flush=True)
    sector_targets = compute_sector_relative_target(raw_data)
    for sector, tickers in sorted({v: [k for k, v2 in SECTOR_MAP.items() if v2 == v and k in raw_data] for v in set(SECTOR_MAP.values())}.items()):
        if tickers:
            print(f"  {sector}: {tickers}", flush=True)

    # Process each ticker and also build a combined dataset
    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []
    combined_feature_names = None

    for ticker, df in raw_data.items():
        print(f"\nProcessing {ticker}...", flush=True)

        # Compute features and sector-relative target
        features_df = compute_all_features(df, ticker=ticker)
        target = sector_targets.get(ticker)
        if target is None:
            print(f"  WARNING: No sector-relative target for {ticker}, skipping", flush=True)
            continue

        # Align and drop NaN rows (features have warmup periods)
        combined = pd.concat([features_df, target.rename("target")], axis=1).dropna()

        # Filter to event window (days near earnings announcements)
        earnings_dates = get_earnings_dates(ticker)
        all_days = len(combined)
        combined = filter_event_window(combined, earnings_dates, EVENT_WINDOW_DAYS)
        event_days = len(combined)

        feature_names = [c for c in combined.columns if c != "target"]
        X = combined[feature_names].values.astype(np.float32)
        y = combined["target"].values.astype(np.float32)
        dates = combined.index.values

        if len(X) < 20:
            print(f"  WARNING: Only {len(X)} event-window samples for {ticker}, skipping", flush=True)
            continue

        print(f"  Features: {len(feature_names)}", flush=True)
        print(f"  All days: {all_days}, Event-window days: {event_days} ({100*event_days/max(all_days,1):.0f}%)", flush=True)
        print(f"  Earnings dates found: {len(earnings_dates)}", flush=True)
        print(f"  Outperform-sector fraction: {y.mean():.3f}", flush=True)

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
            "outperform_fraction_train": float(split["y_train"].mean()),
            "outperform_fraction_val": float(split["y_val"].mean()),
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
            "outperform_fraction_train": float(y_train_all.mean()),
            "outperform_fraction_val": float(y_val_all.mean()),
        }
        (combined_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        print(f"\nCombined dataset: {len(X_train_all)} train, {len(X_val_all)} val samples across {len(all_X_train)} tickers", flush=True)

    print("\nData preparation complete.", flush=True)


if __name__ == "__main__":
    main()
