#!/usr/bin/env python3
"""Fetch balance sheet items (debt, cash) and historical EV/Revenue multiples via FactSet Fundamentals SDK.

Used during projection (Step 8: Trading Signal Assessment) for EV calculation and valuation framework.
Also attempts to fetch short interest metrics with graceful fallback.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _repo_dir() -> Path:
    base = Path(__file__).resolve().parent.parent.parent.parent
    if base.name == ".claude":
        return base.parent
    return base


def _load_local_env() -> None:
    env_path = _repo_dir() / ".env"
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


def _data_dir() -> Path:
    env = os.environ.get("DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    p = _repo_dir()
    for ancestor in [p] + list(p.parents):
        if ancestor.name == "hedge-fund-data":
            return ancestor
    return p.parent / "hedge-fund-data"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_date(raw: Any) -> Optional[date]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if len(text) >= 10:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _factset_ticker(ticker: str, suffix: str) -> str:
    return f"{ticker.upper()}-{suffix.upper()}"


def _unwrap_response(resp: Any) -> Any:
    """Unwrap FactSet SDK response wrappers to get the actual response object."""
    if hasattr(resp, "get_response"):
        return resp.get_response()
    return resp


def _factset_config():
    import fds.sdk.FactSetFundamentals

    user_id = os.environ.get("FACTSET_USER_ID", "").strip()
    api_key = os.environ.get("FACTSET_API_KEY", "").strip()
    if not user_id or not api_key:
        raise SystemExit("Missing FACTSET_USER_ID or FACTSET_API_KEY in env or .env")
    return fds.sdk.FactSetFundamentals.Configuration(
        username=user_id,
        password=api_key,
    )


def _parse_existing_manifest(manifest_path: Path) -> Optional[dict[str, Any]]:
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _has_existing_data(manifest: dict[str, Any]) -> bool:
    downloaded = manifest.get("downloaded_fundamentals")
    if not isinstance(downloaded, list):
        return False
    for item in downloaded:
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("local_path") or "")).expanduser()
        if path.exists() and path.is_file():
            return True
    return False


def _build_fundamentals_request(
    ids: list[str],
    metrics: list[str],
    periodicity: str,
    fiscal_start: str,
    fiscal_end: str,
    currency: str = "USD",
):
    """Build a FundamentalsRequest with proper nested model structure."""
    from fds.sdk.FactSetFundamentals.model.fundamentals_request import FundamentalsRequest
    from fds.sdk.FactSetFundamentals.model.fundamental_request_body import FundamentalRequestBody
    from fds.sdk.FactSetFundamentals.model.fiscal_period import FiscalPeriod
    from fds.sdk.FactSetFundamentals.model.ids_batch_max30000 import IdsBatchMax30000
    from fds.sdk.FactSetFundamentals.model.metrics import Metrics
    from fds.sdk.FactSetFundamentals.model.periodicity import Periodicity

    body = FundamentalRequestBody(
        ids=IdsBatchMax30000(ids),
        metrics=Metrics(metrics),
        periodicity=Periodicity(periodicity),
        fiscal_period=FiscalPeriod(start=fiscal_start, end=fiscal_end),
        currency=currency,
    )
    return FundamentalsRequest(data=body)


def _fetch_balance_sheet(
    api_client,
    ticker_fs: str,
    as_of_date: date,
    lookback_quarters: int = 4,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch balance sheet items: total debt, cash & equivalents."""
    from fds.sdk.FactSetFundamentals.api.fact_set_fundamentals_api import FactSetFundamentalsApi

    api_instance = FactSetFundamentalsApi(api_client)
    api_url = "https://api.factset.com/content/factset-fundamentals/v2/fundamentals"

    metrics = [
        "FF_DEBT",       # Total debt
        "FF_CASH_ST",    # Cash and short-term investments
        "FF_ASSETS",     # Total assets (for context)
    ]

    start_date = f"{as_of_date.year - 2}-01-01"
    end_date = str(as_of_date)

    request = _build_fundamentals_request(
        ids=[ticker_fs],
        metrics=metrics,
        periodicity="QTR",
        fiscal_start=start_date,
        fiscal_end=end_date,
    )
    wrapper = api_instance.get_fds_fundamentals_for_list(fundamentals_request=request)
    response = _unwrap_response(wrapper)

    records: list[dict[str, Any]] = []
    if hasattr(response, "data") and response.data:
        for item in response.data:
            item_dict = item.to_dict() if hasattr(item, "to_dict") else {}
            records.append(item_dict)

    return records, api_url


def _fetch_ev_revenue_history(
    api_client,
    ticker_fs: str,
    as_of_date: date,
    lookback_years: int = 5,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch historical EV/Revenue multiples (quarterly) for lookback period."""
    from fds.sdk.FactSetFundamentals.api.fact_set_fundamentals_api import FactSetFundamentalsApi

    api_instance = FactSetFundamentalsApi(api_client)
    api_url = "https://api.factset.com/content/factset-fundamentals/v2/fundamentals"

    # FF_ENTRPR_VAL may be null for small-caps; fetch components to compute EV/Revenue
    metrics = [
        "FF_MKT_VAL",    # Market capitalization
        "FF_DEBT",        # Total debt
        "FF_CASH_ST",     # Cash & short-term investments
        "FF_SALES",       # Revenue (LTM)
    ]

    start_year = as_of_date.year - lookback_years
    start_date = f"{start_year}-01-01"
    end_date = str(as_of_date)

    request = _build_fundamentals_request(
        ids=[ticker_fs],
        metrics=metrics,
        periodicity="LTM",
        fiscal_start=start_date,
        fiscal_end=end_date,
    )
    wrapper = api_instance.get_fds_fundamentals_for_list(fundamentals_request=request)
    response = _unwrap_response(wrapper)

    records: list[dict[str, Any]] = []
    if hasattr(response, "data") and response.data:
        for item in response.data:
            item_dict = item.to_dict() if hasattr(item, "to_dict") else {}
            records.append(item_dict)

    return records, api_url


def _fetch_short_interest(
    api_client,
    ticker_fs: str,
    as_of_date: date,
) -> tuple[Optional[list[dict[str, Any]]], str]:
    """Attempt to fetch short interest metrics. Returns None on failure (graceful)."""
    from fds.sdk.FactSetFundamentals.api.fact_set_fundamentals_api import FactSetFundamentalsApi

    api_instance = FactSetFundamentalsApi(api_client)
    api_url = "https://api.factset.com/content/factset-fundamentals/v2/fundamentals"

    # Short interest metrics (may not be available for all companies)
    metrics = [
        "FF_SHS_SHORT",         # Shares short
        "FF_SHORT_INT_RATIO",   # Short interest ratio (days to cover)
    ]

    try:
        request = _build_fundamentals_request(
            ids=[ticker_fs],
            metrics=metrics,
            periodicity="QTR",
            fiscal_start=f"{as_of_date.year - 1}-01-01",
            fiscal_end=str(as_of_date),
        )
        wrapper = api_instance.get_fds_fundamentals_for_list(fundamentals_request=request)
        response = _unwrap_response(wrapper)
    except Exception:
        return None, api_url

    records: list[dict[str, Any]] = []
    if hasattr(response, "data") and response.data:
        for item in response.data:
            item_dict = item.to_dict() if hasattr(item, "to_dict") else {}
            records.append(item_dict)

    return records if records else None, api_url


def main() -> None:
    _load_local_env()

    parser = argparse.ArgumentParser(
        description="Fetch balance sheet, EV/Revenue history, and short interest via FactSet Fundamentals SDK."
    )
    parser.add_argument("--company-id", default="ECOR", help="Company runtime id (default: ECOR)")
    parser.add_argument("--ticker", required=True, help="Ticker symbol")
    parser.add_argument("--cutoff-date", required=True, help="Cutoff date YYYY-MM-DD")
    parser.add_argument(
        "--exchange-suffix",
        default=os.environ.get("FACTSET_EXCHANGE_SUFFIX", "US"),
        help="FactSet exchange suffix (default: US)",
    )
    parser.add_argument(
        "--lookback-years",
        type=int,
        default=5,
        help="Years of EV/Revenue history to fetch (default: 5)",
    )
    parser.add_argument(
        "--skip-if-present",
        action="store_true",
        help="Skip retrieval if manifest already has existing data.",
    )
    args = parser.parse_args()

    cutoff = _parse_date(args.cutoff_date)
    if cutoff is None:
        raise SystemExit(f"Invalid cutoff date: {args.cutoff_date}")

    import fds.sdk.FactSetFundamentals

    company_id = args.company_id.upper()
    ticker = args.ticker.upper()
    ticker_fs = _factset_ticker(ticker, args.exchange_suffix)

    run_cache = _data_dir() / "companies" / company_id / "cache" / f"asof_{args.cutoff_date}"
    fundamentals_dir = run_cache / "fundamentals" / ticker
    fundamentals_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = fundamentals_dir / "fetch_manifest.json"

    if args.skip_if_present:
        existing = _parse_existing_manifest(manifest_path)
        if isinstance(existing, dict) and _has_existing_data(existing):
            print(f"Skipping FactSet fundamentals fetch; existing data found in {manifest_path}")
            return

    configuration = _factset_config()
    downloaded_fundamentals: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    with fds.sdk.FactSetFundamentals.ApiClient(configuration) as api_client:
        # 1. Fetch balance sheet items
        try:
            bs_records, bs_url = _fetch_balance_sheet(api_client, ticker_fs, cutoff)
            bs_path = fundamentals_dir / "balance_sheet.json"
            bs_path.write_text(json.dumps(bs_records, indent=2, default=str) + "\n")
            downloaded_fundamentals.append({
                "source_type": "fundamental",
                "retrieval_method": "factset_fundamentals_api",
                "data_type": "balance_sheet",
                "ticker": ticker,
                "factset_ticker": ticker_fs,
                "url": bs_url,
                "local_path": str(bs_path),
                "records_count": len(bs_records),
                "published_date": args.cutoff_date,
            })
            print(f"Balance sheet: {len(bs_records)} records")
        except Exception as exc:
            errors.append({
                "step": "balance_sheet",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"Error fetching balance sheet: {exc}")

        # 2. Fetch historical EV/Revenue multiples
        try:
            ev_records, ev_url = _fetch_ev_revenue_history(
                api_client, ticker_fs, cutoff, args.lookback_years
            )
            ev_path = fundamentals_dir / "ev_revenue_history.json"
            ev_path.write_text(json.dumps(ev_records, indent=2, default=str) + "\n")
            downloaded_fundamentals.append({
                "source_type": "fundamental",
                "retrieval_method": "factset_fundamentals_api",
                "data_type": "ev_revenue_history",
                "ticker": ticker,
                "factset_ticker": ticker_fs,
                "url": ev_url,
                "local_path": str(ev_path),
                "records_count": len(ev_records),
                "published_date": args.cutoff_date,
                "lookback_years": args.lookback_years,
            })
            print(f"EV/Revenue history: {len(ev_records)} records ({args.lookback_years}yr lookback)")
        except Exception as exc:
            errors.append({
                "step": "ev_revenue_history",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"Error fetching EV/Revenue history: {exc}")

        # 3. Fetch short interest (graceful fallback)
        try:
            si_records, si_url = _fetch_short_interest(api_client, ticker_fs, cutoff)
            if si_records is not None:
                si_path = fundamentals_dir / "short_interest.json"
                si_path.write_text(json.dumps(si_records, indent=2, default=str) + "\n")
                downloaded_fundamentals.append({
                    "source_type": "fundamental",
                    "retrieval_method": "factset_fundamentals_api",
                    "data_type": "short_interest",
                    "ticker": ticker,
                    "factset_ticker": ticker_fs,
                    "url": si_url,
                    "local_path": str(si_path),
                    "records_count": len(si_records),
                    "published_date": args.cutoff_date,
                })
                print(f"Short interest: {len(si_records)} records")
            else:
                print("Short interest: not available (graceful fallback)")
        except Exception as exc:
            # Graceful fallback — short interest is optional
            print(f"Short interest not available: {exc}")

    manifest = {
        "created_at_utc": _now_iso(),
        "company_id": company_id,
        "ticker": ticker,
        "factset_ticker": ticker_fs,
        "cutoff_date": args.cutoff_date,
        "lookback_years": args.lookback_years,
        "downloaded_fundamentals_count": len(downloaded_fundamentals),
        "downloaded_fundamentals": downloaded_fundamentals,
        "errors": errors,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Downloaded fundamental files: {len(downloaded_fundamentals)}")

    if not downloaded_fundamentals:
        raise SystemExit(
            "No FactSet fundamentals data was downloaded. "
            "Check credentials, ticker, and cutoff date."
        )


if __name__ == "__main__":
    main()
