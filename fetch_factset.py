#!/usr/bin/env python3
"""Fetch and cache FactSet fundamental data for all tickers in the universe.

Run this once before training. Fetches:
- Consensus revenue estimates (rolling, for revision momentum)
- Short interest
- Quarterly actuals (for earnings surprise history)

Caches to DATA_DIR/{ticker}/factset_cache.json. prepare.py loads from cache.

Usage:
    python fetch_factset.py                     # Fetch for default universe
    TICKERS=AAPL,MSFT python fetch_factset.py   # Specific tickers
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# ETFs that won't have FactSet fundamental data
ETFS = {"SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI",
        "XLP", "XLU", "XLB", "XLC", "XLRE"}

DEFAULT_TICKERS = "AAPL,MSFT,AMZN,GOOGL,META,NVDA,TSLA,CRM,ADBE,ORCL,INTC,AMD,CSCO,AVGO,QCOM,TXN,NOW,INTU,AMAT,MU,JPM,BAC,GS,MS,V,MA,WFC,C,AXP,BLK,SCHW,USB,PNC,TFC,COF,JNJ,UNH,PFE,ABT,TMO,ABBV,MRK,LLY,MDT,DHR,BMY,AMGN,GILD,ISRG,SYK,HD,MCD,NKE,SBUX,TJX,LOW,TGT,ROST,MAR,YUM,WMT,PG,KO,PEP,COST,CL,MO,GIS,CAT,BA,UPS,HON,UNP,RTX,DE,GE,LMT,MMM,XOM,CVX,COP,SLB,EOG,MPC,NEE,DUK,SO,D,AMT,PLD,CCI,EQIX,LIN,APD,SHW,ECL,DIS,CMCSA,NFLX,T"
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))


def _load_env() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _factset_ticker(ticker: str) -> str:
    return f"{ticker.upper()}-US"


def _fetch_consensus_time_series(ticker: str, start_date: date, end_date: date) -> list[dict]:
    """Fetch rolling consensus revenue estimates at monthly intervals.

    Returns list of {date, estimate, num_analysts, fiscal_period} dicts.
    """
    import fds.sdk.FactSetEstimates
    from fds.sdk.FactSetEstimates.api import consensus_api

    config = fds.sdk.FactSetEstimates.Configuration(
        username=os.environ.get("FACTSET_USER_ID", "").strip(),
        password=os.environ.get("FACTSET_API_KEY", "").strip(),
    )

    records = []
    ticker_fs = _factset_ticker(ticker)

    with fds.sdk.FactSetEstimates.ApiClient(config) as client:
        api = consensus_api.ConsensusApi(client)

        # Sample at monthly intervals to build revision time series
        sample_date = start_date
        while sample_date <= end_date:
            try:
                # Get consensus for next quarter
                q = (sample_date.month - 1) // 3 + 1
                y = sample_date.year
                # Next quarter
                nq = q + 1 if q < 4 else 1
                ny = y if q < 4 else y + 1
                fp = f"{ny}/{nq}F"

                response = api.get_fixed_consensus(
                    ids=[ticker_fs],
                    metrics=["SALES"],
                    start_date=sample_date,
                    end_date=sample_date,
                    frequency="D",
                    fiscal_period_start=fp,
                    fiscal_period_end=fp,
                    periodicity="QTR",
                    currency="USD",
                )

                if hasattr(response, "data") and response.data:
                    for item in response.data:
                        d = item.to_dict() if hasattr(item, "to_dict") else {}
                        estimate = d.get("mean") or d.get("median")
                        n_analysts = d.get("estimate_count") or d.get("analyst_count")
                        if estimate is not None:
                            records.append({
                                "date": str(sample_date),
                                "estimate": float(estimate),
                                "num_analysts": int(n_analysts) if n_analysts else None,
                                "fiscal_period": d.get("fiscal_period", fp),
                            })
            except Exception as e:
                print(f"    Warning: consensus fetch failed for {ticker} on {sample_date}: {e}", flush=True)

            sample_date += timedelta(days=30)

    return records


def _fetch_actuals(ticker: str, start_date: date, end_date: date) -> list[dict]:
    """Fetch quarterly revenue actuals for earnings surprise computation."""
    import fds.sdk.FactSetEstimates
    from fds.sdk.FactSetEstimates.api import consensus_api

    config = fds.sdk.FactSetEstimates.Configuration(
        username=os.environ.get("FACTSET_USER_ID", "").strip(),
        password=os.environ.get("FACTSET_API_KEY", "").strip(),
    )

    records = []
    ticker_fs = _factset_ticker(ticker)

    with fds.sdk.FactSetEstimates.ApiClient(config) as client:
        api = consensus_api.ConsensusApi(client)

        try:
            y_start = start_date.year
            y_end = end_date.year
            fp_start = f"{y_start}/1F"
            fp_end = f"{y_end}/4F"

            response = api.get_fixed_consensus(
                ids=[ticker_fs],
                metrics=["SALES"],
                start_date=end_date,
                end_date=end_date,
                frequency="D",
                fiscal_period_start=fp_start,
                fiscal_period_end=fp_end,
                periodicity="QTR",
                currency="USD",
            )

            if hasattr(response, "data") and response.data:
                for item in response.data:
                    d = item.to_dict() if hasattr(item, "to_dict") else {}
                    records.append({
                        "fiscal_period": d.get("fiscal_period"),
                        "estimate": float(d["mean"]) if d.get("mean") else None,
                        "actual": float(d["actual"]) if d.get("actual") else None,
                        "surprise_pct": float((d["actual"] - d["mean"]) / d["mean"] * 100)
                            if d.get("actual") and d.get("mean") and d["mean"] != 0 else None,
                        "report_date": str(d.get("report_date") or ""),
                    })
        except Exception as e:
            print(f"    Warning: actuals fetch failed for {ticker}: {e}", flush=True)

    return records


def _fetch_short_interest(ticker: str, as_of_date: date) -> Optional[dict]:
    """Fetch most recent short interest data."""
    import fds.sdk.FactSetFundamentals
    from fds.sdk.FactSetFundamentals.api.fact_set_fundamentals_api import FactSetFundamentalsApi
    from fds.sdk.FactSetFundamentals.model.fundamentals_request import FundamentalsRequest
    from fds.sdk.FactSetFundamentals.model.fundamental_request_body import FundamentalRequestBody
    from fds.sdk.FactSetFundamentals.model.fiscal_period import FiscalPeriod
    from fds.sdk.FactSetFundamentals.model.ids_batch_max30000 import IdsBatchMax30000
    from fds.sdk.FactSetFundamentals.model.metrics import Metrics
    from fds.sdk.FactSetFundamentals.model.periodicity import Periodicity

    config = fds.sdk.FactSetFundamentals.Configuration(
        username=os.environ.get("FACTSET_USER_ID", "").strip(),
        password=os.environ.get("FACTSET_API_KEY", "").strip(),
    )

    ticker_fs = _factset_ticker(ticker)

    try:
        with fds.sdk.FactSetFundamentals.ApiClient(config) as client:
            api = FactSetFundamentalsApi(client)

            body = FundamentalRequestBody(
                ids=IdsBatchMax30000([ticker_fs]),
                metrics=Metrics(["FF_SHS_SHORT", "FF_SHORT_INT_RATIO"]),
                periodicity=Periodicity("QTR"),
                fiscal_period=FiscalPeriod(
                    start=f"{as_of_date.year - 1}-01-01",
                    end=str(as_of_date),
                ),
                currency="USD",
            )
            request = FundamentalsRequest(data=body)
            wrapper = api.get_fds_fundamentals_for_list(fundamentals_request=request)

            response = wrapper.get_response() if hasattr(wrapper, "get_response") else wrapper
            if hasattr(response, "data") and response.data:
                # Take most recent record
                latest = response.data[-1]
                d = latest.to_dict() if hasattr(latest, "to_dict") else {}
                return {
                    "shares_short": d.get("value") if d.get("metric") == "FF_SHS_SHORT" else None,
                    "short_interest_ratio": d.get("value") if d.get("metric") == "FF_SHORT_INT_RATIO" else None,
                    "date": str(d.get("date") or as_of_date),
                }
    except Exception as e:
        print(f"    Warning: short interest fetch failed for {ticker}: {e}", flush=True)

    return None


def main() -> None:
    _load_env()

    user_id = os.environ.get("FACTSET_USER_ID", "").strip()
    api_key = os.environ.get("FACTSET_API_KEY", "").strip()
    if not user_id or not api_key:
        print("ERROR: Set FACTSET_USER_ID and FACTSET_API_KEY in .env", flush=True)
        sys.exit(1)

    tickers_str = os.environ.get("TICKERS", DEFAULT_TICKERS)
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * 8)  # 8 years of history

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        cache_path = DATA_DIR / ticker.lower() / "factset_cache.json"

        if cache_path.exists():
            print(f"{ticker}: cached (skip)", flush=True)
            continue

        if ticker in ETFS:
            print(f"{ticker}: ETF (skip — no fundamental data)", flush=True)
            # Write empty cache so we don't retry
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"ticker": ticker, "is_etf": True}, indent=2))
            continue

        print(f"{ticker}: fetching FactSet data...", flush=True)

        # Consensus time series
        print(f"  Fetching consensus estimates...", flush=True)
        consensus = _fetch_consensus_time_series(ticker, start_date, end_date)
        print(f"  Got {len(consensus)} consensus records", flush=True)

        # Actuals
        print(f"  Fetching actuals...", flush=True)
        actuals = _fetch_actuals(ticker, start_date, end_date)
        print(f"  Got {len(actuals)} actuals records", flush=True)

        # Save cache
        cache_data = {
            "ticker": ticker,
            "is_etf": False,
            "fetched_at": datetime.now().isoformat(),
            "consensus_time_series": consensus,
            "actuals": actuals,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache_data, indent=2) + "\n")
        print(f"  Cached to {cache_path}", flush=True)

    print("\nFactSet data fetch complete.", flush=True)


if __name__ == "__main__":
    main()
