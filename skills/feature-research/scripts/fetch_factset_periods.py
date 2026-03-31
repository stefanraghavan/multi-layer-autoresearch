#!/usr/bin/env python3
"""Discover backtest period schedule from FactSet Actuals API.

Calls ActualsApi.get_actuals() to fetch historical quarterly revenue with
report dates and fiscal end dates.  This replaces the manual CSV for period
schedule discovery, enabling automatic scaling to any company.

Output:
  ~/hedge-fund-data/companies/{COMPANY}/cache/periods_discovery.json
  ~/hedge-fund-data/companies/{COMPANY}/cache/periods_discovery_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional


def _repo_dir() -> Path:
    """Repo or company-runtime root.

    From repo:    scripts/ -> earnings-projection/ -> skills/ -> REPO (4 parents)
    From runtime: scripts/ -> earnings-projection/ -> skills/ -> .claude/ -> COMPANY (5 parents)
    """
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
            line = line[len("export ") :].strip()
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


def _factset_config():
    import fds.sdk.FactSetEstimates

    user_id = os.environ.get("FACTSET_USER_ID", "").strip()
    api_key = os.environ.get("FACTSET_API_KEY", "").strip()
    if not user_id or not api_key:
        raise SystemExit("Missing FACTSET_USER_ID or FACTSET_API_KEY in env or .env")
    return fds.sdk.FactSetEstimates.Configuration(
        username=user_id,
        password=api_key,
    )


def _fetch_actuals(
    api_client,
    ticker_fs: str,
    lookback_quarters: int,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch reported actuals with report dates and fiscal end dates."""
    from fds.sdk.FactSetEstimates.api import actuals_api

    api_instance = actuals_api.ActualsApi(api_client)
    api_url = "https://api.factset.com/content/factset-estimates/v2/actuals"

    response = api_instance.get_actuals(
        ids=[ticker_fs],
        metrics=["SALES"],
        relative_fiscal_start=-lookback_quarters,
        relative_fiscal_end=0,
        periodicity="QTR",
        currency="USD",
    )

    records: list[dict[str, Any]] = []
    if hasattr(response, "data") and response.data:
        for item in response.data:
            item_dict = item.to_dict() if hasattr(item, "to_dict") else {}
            records.append(item_dict)

    return records, api_url


def _transform_to_period_rows(
    records: list[dict[str, Any]],
    cutoff_days: int,
) -> list[dict[str, Any]]:
    """Transform FactSet Actuals API records into period rows.

    Each record is expected to have (camelCase keys from to_dict()):
        fiscalPeriod (int 1-4), fiscalYear (int), reportDate (date/str),
        fiscalEndDate (date/str), actualValue (float).
    """
    rows: list[dict[str, Any]] = []
    seen_periods: set[str] = set()

    for rec in records:
        fp = rec.get("fiscalPeriod") or rec.get("fiscal_period")
        fy = rec.get("fiscalYear") or rec.get("fiscal_year")
        report_date = _parse_date(rec.get("reportDate") or rec.get("report_date"))
        fiscal_end = _parse_date(rec.get("fiscalEndDate") or rec.get("fiscal_end_date"))
        actual_value = rec.get("actualValue") or rec.get("actual_value") or rec.get("value")

        if fp is None or fy is None:
            continue
        if report_date is None:
            continue
        if actual_value is None:
            continue

        try:
            fp_int = int(fp)
            fy_int = int(fy)
            actual_float = float(actual_value)
        except (ValueError, TypeError):
            continue

        if fp_int < 1 or fp_int > 4:
            continue

        period = f"Q{fp_int}_{fy_int}"
        if period in seen_periods:
            continue
        seen_periods.add(period)

        cutoff = (report_date - timedelta(days=cutoff_days)).isoformat()

        row: dict[str, Any] = {
            "period": period,
            "cutoff": cutoff,
            "actual_m": actual_float,
            "consensus_m": None,
            "earnings_date": report_date.isoformat(),
        }
        if fiscal_end is not None:
            row["target_period_end_date"] = fiscal_end.isoformat()

        rows.append(row)

    rows.sort(key=lambda r: r["earnings_date"])
    return rows


def main() -> None:
    _load_local_env()

    parser = argparse.ArgumentParser(
        description="Discover backtest period schedule from FactSet Actuals API."
    )
    parser.add_argument("--company-id", default="ECOR", help="Company runtime id (default: ECOR)")
    parser.add_argument("--ticker", required=True, help="Ticker symbol")
    parser.add_argument(
        "--exchange-suffix",
        default=os.environ.get("FACTSET_EXCHANGE_SUFFIX", "US"),
        help="FactSet exchange suffix (default: US)",
    )
    parser.add_argument(
        "--lookback-quarters",
        type=int,
        default=24,
        help="Number of historical quarters to fetch (default: 24)",
    )
    parser.add_argument(
        "--cutoff-days-before-earnings",
        type=int,
        default=int(os.environ.get("CUTOFF_DAYS_BEFORE_EARNINGS", "7")),
        help="Days before earnings date to set cutoff (default: 7)",
    )
    parser.add_argument(
        "--skip-if-present",
        action="store_true",
        help="Skip retrieval if periods_discovery.json already exists.",
    )
    args = parser.parse_args()

    import fds.sdk.FactSetEstimates

    company_id = args.company_id.upper()
    ticker = args.ticker.upper()
    ticker_fs = _factset_ticker(ticker, args.exchange_suffix)

    company_cache = _data_dir() / "companies" / company_id / "cache"
    company_cache.mkdir(parents=True, exist_ok=True)
    discovery_path = company_cache / "periods_discovery.json"
    manifest_path = company_cache / "periods_discovery_manifest.json"

    if args.skip_if_present and discovery_path.exists():
        print(f"Skipping period discovery; existing data found at {discovery_path}")
        return

    configuration = _factset_config()

    with fds.sdk.FactSetEstimates.ApiClient(configuration) as api_client:
        records, api_url = _fetch_actuals(api_client, ticker_fs, args.lookback_quarters)

    print(f"FactSet Actuals API returned {len(records)} records for {ticker_fs}")

    rows = _transform_to_period_rows(records, args.cutoff_days_before_earnings)
    print(f"Transformed into {len(rows)} period rows")

    if not rows:
        raise SystemExit(
            "No valid period rows could be built from FactSet Actuals. "
            "Check ticker, credentials, and that the company has quarterly revenue data."
        )

    # Log first and last periods for operator verification
    print(f"Period range: {rows[0]['period']} -> {rows[-1]['period']}")
    for row in rows[:3]:
        print(f"  {row['period']}: actual_m={row['actual_m']}, cutoff={row['cutoff']}, earnings={row['earnings_date']}")
    if len(rows) > 3:
        print(f"  ... ({len(rows) - 3} more)")

    discovery_path.write_text(json.dumps(rows, indent=2, default=str) + "\n")
    print(f"Wrote period discovery: {discovery_path}")

    manifest = {
        "created_at_utc": _now_iso(),
        "company_id": company_id,
        "ticker": ticker,
        "factset_ticker": ticker_fs,
        "api_url": api_url,
        "lookback_quarters": args.lookback_quarters,
        "cutoff_days_before_earnings": args.cutoff_days_before_earnings,
        "raw_records_count": len(records),
        "period_rows_count": len(rows),
        "period_range": f"{rows[0]['period']} -> {rows[-1]['period']}",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
