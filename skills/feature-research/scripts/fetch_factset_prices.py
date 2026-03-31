#!/usr/bin/env python3
"""Fetch stock prices, market value, shares outstanding, and YTD returns via FactSet Global Prices SDK.

Used during projection (Step 8: Trading Signal Assessment) and backtesting.
Fetches as-of cutoff date for the company and its peer group.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone
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
    import fds.sdk.FactSetGlobalPrices

    user_id = os.environ.get("FACTSET_USER_ID", "").strip()
    api_key = os.environ.get("FACTSET_API_KEY", "").strip()
    if not user_id or not api_key:
        raise SystemExit("Missing FACTSET_USER_ID or FACTSET_API_KEY in env or .env")
    return fds.sdk.FactSetGlobalPrices.Configuration(
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
    downloaded = manifest.get("downloaded_prices")
    if not isinstance(downloaded, list):
        return False
    for item in downloaded:
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("local_path") or "")).expanduser()
        if path.exists() and path.is_file():
            return True
    return False


def _fetch_prices(
    api_client,
    ticker_fs: str,
    as_of_date: date,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch close price, market cap, and shares outstanding as of a date."""
    from fds.sdk.FactSetGlobalPrices.api import prices_api
    from fds.sdk.FactSetGlobalPrices.model.global_prices_request import GlobalPricesRequest
    from fds.sdk.FactSetGlobalPrices.model.ids_batch_max2000 import IdsBatchMax2000
    from fds.sdk.FactSetGlobalPrices.model.prices_fields import PricesFields
    from fds.sdk.FactSetGlobalPrices.model.frequency import Frequency

    api_instance = prices_api.PricesApi(api_client)
    api_url = "https://api.factset.com/content/factset-global-prices/v1/prices"

    request = GlobalPricesRequest(
        ids=IdsBatchMax2000([ticker_fs]),
        fields=PricesFields(["price", "volume"]),
        start_date=str(as_of_date),
        end_date=str(as_of_date),
        frequency=Frequency("D"),
        currency="USD",
    )

    wrapper = api_instance.get_security_prices_for_list(global_prices_request=request)
    response = _unwrap_response(wrapper)

    records: list[dict[str, Any]] = []
    if hasattr(response, "data") and response.data:
        for item in response.data:
            item_dict = item.to_dict() if hasattr(item, "to_dict") else {}
            records.append(item_dict)

    return records, api_url


def _fetch_market_value(
    ticker_fs_list: list[str] | str,
    as_of_date: date,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch market cap and shares outstanding via FactSet Fundamentals SDK.

    Accepts a single ticker string or a list of tickers (company + peers).
    """
    import fds.sdk.FactSetFundamentals
    from fds.sdk.FactSetFundamentals.api.fact_set_fundamentals_api import FactSetFundamentalsApi
    from fds.sdk.FactSetFundamentals.model.fundamentals_request import FundamentalsRequest
    from fds.sdk.FactSetFundamentals.model.fundamental_request_body import FundamentalRequestBody
    from fds.sdk.FactSetFundamentals.model.fiscal_period import FiscalPeriod
    from fds.sdk.FactSetFundamentals.model.ids_batch_max30000 import IdsBatchMax30000
    from fds.sdk.FactSetFundamentals.model.metrics import Metrics
    from fds.sdk.FactSetFundamentals.model.periodicity import Periodicity

    api_url = "https://api.factset.com/content/factset-fundamentals/v2/fundamentals"

    if isinstance(ticker_fs_list, str):
        ticker_fs_list = [ticker_fs_list]

    user_id = os.environ.get("FACTSET_USER_ID", "").strip()
    api_key = os.environ.get("FACTSET_API_KEY", "").strip()
    config = fds.sdk.FactSetFundamentals.Configuration(
        username=user_id, password=api_key,
    )

    # Use date-based fiscal period (API requires YYYY-MM-DD format)
    start_date = f"{as_of_date.year}-01-01"
    end_date = str(as_of_date)

    body = FundamentalRequestBody(
        ids=IdsBatchMax30000(ticker_fs_list),
        metrics=Metrics(["FF_MKT_VAL", "FF_COM_SHS_OUT"]),
        periodicity=Periodicity("QTR"),
        fiscal_period=FiscalPeriod(start=start_date, end=end_date),
        currency="USD",
    )
    request = FundamentalsRequest(data=body)

    with fds.sdk.FactSetFundamentals.ApiClient(config) as client:
        api_instance = FactSetFundamentalsApi(client)
        wrapper = api_instance.get_fds_fundamentals_for_list(fundamentals_request=request)
        response = _unwrap_response(wrapper)

    records: list[dict[str, Any]] = []
    if hasattr(response, "data") and response.data:
        for item in response.data:
            item_dict = item.to_dict() if hasattr(item, "to_dict") else {}
            records.append(item_dict)

    return records, api_url


def _fetch_returns(
    api_client,
    ticker_fs_list: list[str],
    as_of_date: date,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch YTD total returns for a list of tickers."""
    from fds.sdk.FactSetGlobalPrices.api import returns_api
    from fds.sdk.FactSetGlobalPrices.model.returns_request import ReturnsRequest
    from fds.sdk.FactSetGlobalPrices.model.ids_max1000 import IdsMax1000
    from fds.sdk.FactSetGlobalPrices.model.frequency import Frequency

    api_instance = returns_api.ReturnsApi(api_client)
    api_url = "https://api.factset.com/content/factset-global-prices/v1/returns"

    # YTD: from Jan 1 of the same year to cutoff
    ytd_start = date(as_of_date.year, 1, 1)

    request = ReturnsRequest(
        ids=IdsMax1000(ticker_fs_list),
        start_date=str(ytd_start),
        end_date=str(as_of_date),
        frequency=Frequency("D"),
        currency="USD",
    )

    wrapper = api_instance.get_returns_for_list(returns_request=request)
    response = _unwrap_response(wrapper)

    records: list[dict[str, Any]] = []
    if hasattr(response, "data") and response.data:
        for item in response.data:
            item_dict = item.to_dict() if hasattr(item, "to_dict") else {}
            records.append(item_dict)

    return records, api_url


def main() -> None:
    _load_local_env()

    parser = argparse.ArgumentParser(
        description="Fetch stock prices, market value, and YTD returns via FactSet Global Prices SDK."
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
        "--peer-tickers",
        default="",
        help="Comma-separated peer tickers in FactSet format (e.g., SWAV-US,NUVB-US)",
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

    import fds.sdk.FactSetGlobalPrices

    company_id = args.company_id.upper()
    ticker = args.ticker.upper()
    ticker_fs = _factset_ticker(ticker, args.exchange_suffix)

    peer_tickers: list[str] = []
    if args.peer_tickers.strip():
        peer_tickers = [t.strip() for t in args.peer_tickers.split(",") if t.strip()]

    run_cache = _data_dir() / "companies" / company_id / "cache" / f"asof_{args.cutoff_date}"
    prices_dir = run_cache / "prices" / ticker
    prices_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = prices_dir / "fetch_manifest.json"

    if args.skip_if_present:
        existing = _parse_existing_manifest(manifest_path)
        if isinstance(existing, dict) and _has_existing_data(existing):
            print(f"Skipping FactSet prices fetch; existing data found in {manifest_path}")
            return

    configuration = _factset_config()
    downloaded_prices: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    price_records: list[dict[str, Any]] = []

    with fds.sdk.FactSetGlobalPrices.ApiClient(configuration) as api_client:
        # 1. Fetch stock price and market value for the company
        try:
            price_records, price_url = _fetch_prices(api_client, ticker_fs, cutoff)
            price_path = prices_dir / "prices.json"
            price_path.write_text(json.dumps(price_records, indent=2, default=str) + "\n")
            downloaded_prices.append({
                "source_type": "price",
                "retrieval_method": "factset_global_prices_api",
                "data_type": "prices",
                "ticker": ticker,
                "factset_ticker": ticker_fs,
                "url": price_url,
                "local_path": str(price_path),
                "records_count": len(price_records),
                "published_date": args.cutoff_date,
            })
            print(f"Prices: {len(price_records)} records")
        except Exception as exc:
            errors.append({
                "step": "prices",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"Error fetching prices: {exc}")

        # 1b. Fetch market value and shares outstanding for company + peers
        all_mv_tickers = [ticker_fs] + peer_tickers
        try:
            mv_records, mv_url = _fetch_market_value(all_mv_tickers, cutoff)
            # Build market_value.json with price from step 1a if available
            stock_price = None
            if price_records:
                stock_price = price_records[0].get("price")
            market_value_data = []
            for rec in mv_records:
                rec_id = rec.get("request_id") or rec.get("requestId") or ""
                mv = rec.get("ff_mkt_val") or rec.get("FF_MKT_VAL")
                so = rec.get("ff_com_shs_out") or rec.get("FF_COM_SHS_OUT")
                entry = {
                    "request_id": rec_id,
                    "date": str(cutoff),
                    "market_value": mv,
                    "shares_outstanding": so,
                }
                # Include stock price only for the primary ticker
                if rec_id.upper().startswith(ticker.upper()):
                    entry["price"] = stock_price
                market_value_data.append(entry)
            mv_path = prices_dir / "market_value.json"
            mv_path.write_text(json.dumps(market_value_data, indent=2, default=str) + "\n")
            downloaded_prices.append({
                "source_type": "price",
                "retrieval_method": "factset_fundamentals_api",
                "data_type": "market_value",
                "tickers": all_mv_tickers,
                "url": mv_url,
                "local_path": str(mv_path),
                "records_count": len(market_value_data),
            })
            print(f"Market value records: {len(market_value_data)} for {len(all_mv_tickers)} tickers")
        except Exception as exc:
            errors.append({
                "step": "market_value",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"Error fetching market value: {exc}")

        # 2. Fetch YTD returns for company + peers
        all_return_tickers = [ticker_fs] + peer_tickers
        try:
            return_records, returns_url = _fetch_returns(api_client, all_return_tickers, cutoff)
            returns_path = prices_dir / "returns_ytd.json"
            returns_path.write_text(json.dumps(return_records, indent=2, default=str) + "\n")
            downloaded_prices.append({
                "source_type": "price",
                "retrieval_method": "factset_global_prices_api",
                "data_type": "returns_ytd",
                "tickers": all_return_tickers,
                "url": returns_url,
                "local_path": str(returns_path),
                "records_count": len(return_records),
                "published_date": args.cutoff_date,
            })
            print(f"YTD returns: {len(return_records)} records for {len(all_return_tickers)} tickers")
        except Exception as exc:
            errors.append({
                "step": "returns_ytd",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"Error fetching YTD returns: {exc}")

    manifest = {
        "created_at_utc": _now_iso(),
        "company_id": company_id,
        "ticker": ticker,
        "factset_ticker": ticker_fs,
        "cutoff_date": args.cutoff_date,
        "peer_tickers": peer_tickers,
        "downloaded_prices_count": len(downloaded_prices),
        "downloaded_prices": downloaded_prices,
        "errors": errors,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Downloaded price files: {len(downloaded_prices)}")

    if not downloaded_prices:
        raise SystemExit(
            "No FactSet price data was downloaded. "
            "Check credentials, ticker, and cutoff date."
        )


if __name__ == "__main__":
    main()
