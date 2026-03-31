#!/usr/bin/env python3
"""Fetch consensus and segment consensus estimates via FactSet Estimates SDK into cutoff-scoped cache.

Used during postmortem only — the agent does not call this script during projection.
During projection, historical financials come from SEC EDGAR and guidance from transcripts.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timezone
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


def _parse_existing_manifest(manifest_path: Path) -> Optional[dict[str, Any]]:
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _has_existing_data(manifest: dict[str, Any]) -> bool:
    downloaded = manifest.get("downloaded_estimates")
    if not isinstance(downloaded, list):
        return False
    for item in downloaded:
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("local_path") or "")).expanduser()
        if path.exists() and path.is_file():
            return True
    return False


def _fiscal_period_range(
    cutoff: date,
    lookback_quarters: int,
    forward_quarters: int = 8,
) -> tuple[str, str]:
    """Build fiscal_period_start/end strings like '2020/1F' for the consensus API."""
    # Estimate current fiscal quarter from cutoff date (approximate)
    q = (cutoff.month - 1) // 3 + 1
    y = cutoff.year

    # Go back lookback_quarters
    total_start = (y * 4 + q) - lookback_quarters
    start_y, start_q = divmod(total_start - 1, 4)
    start_q += 1

    # Go forward forward_quarters
    total_end = (y * 4 + q) + forward_quarters
    end_y, end_q = divmod(total_end - 1, 4)
    end_q += 1

    return f"{start_y}/{start_q}F", f"{end_y}/{end_q}F"


def _fetch_consensus(
    api_client,
    ticker_fs: str,
    consensus_date: date,
    lookback_quarters: int,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch consensus revenue estimates as of consensus_date (typically earnings date)."""
    from fds.sdk.FactSetEstimates.api import consensus_api

    api_instance = consensus_api.ConsensusApi(api_client)

    api_url = "https://api.factset.com/content/factset-estimates/v2/fixed-consensus"

    fp_start, fp_end = _fiscal_period_range(consensus_date, lookback_quarters)

    response = api_instance.get_fixed_consensus(
        ids=[ticker_fs],
        metrics=["SALES"],
        start_date=consensus_date,
        end_date=consensus_date,
        frequency="D",
        fiscal_period_start=fp_start,
        fiscal_period_end=fp_end,
        periodicity="QTR",
        currency="USD",
    )

    records: list[dict[str, Any]] = []
    if hasattr(response, "data") and response.data:
        for item in response.data:
            item_dict = item.to_dict() if hasattr(item, "to_dict") else {}
            records.append(item_dict)

    return records, api_url


def _fetch_annual_consensus(
    api_client,
    ticker_fs: str,
    consensus_date: date,
    forward_years: int = 3,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch annual consensus revenue estimates (CFY, NFY, NFY+1)."""
    from fds.sdk.FactSetEstimates.api import consensus_api

    api_instance = consensus_api.ConsensusApi(api_client)
    api_url = "https://api.factset.com/content/factset-estimates/v2/fixed-consensus"

    y = consensus_date.year
    fp_start = f"{y - 1}/1F"
    fp_end = f"{y + forward_years}/4F"

    response = api_instance.get_fixed_consensus(
        ids=[ticker_fs],
        metrics=["SALES"],
        start_date=consensus_date,
        end_date=consensus_date,
        frequency="D",
        fiscal_period_start=fp_start,
        fiscal_period_end=fp_end,
        periodicity="ANN",
        currency="USD",
    )

    records: list[dict[str, Any]] = []
    if hasattr(response, "data") and response.data:
        for item in response.data:
            item_dict = item.to_dict() if hasattr(item, "to_dict") else {}
            records.append(item_dict)

    return records, api_url


def _fetch_segment_estimates(
    api_client,
    ticker_fs: str,
    consensus_date: date,
    lookback_quarters: int,
    segment_type: str,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch quarterly segment consensus estimates (BUS or GEO)."""
    from fds.sdk.FactSetEstimates.api import segments_api

    api_instance = segments_api.SegmentsApi(api_client)

    api_url = "https://api.factset.com/content/factset-estimates/v2/segments"

    fp_start, fp_end = _fiscal_period_range(consensus_date, lookback_quarters)

    response = api_instance.get_segments(
        ids=[ticker_fs],
        metrics=["SALES"],
        segment_type=segment_type,
        periodicity="QTR",
        start_date=consensus_date,
        end_date=consensus_date,
        frequency="D",
        currency="USD",
        relative_fiscal_start=-lookback_quarters,
        relative_fiscal_end=0,
    )

    records: list[dict[str, Any]] = []
    if hasattr(response, "data") and response.data:
        for item in response.data:
            item_dict = item.to_dict() if hasattr(item, "to_dict") else {}
            records.append(item_dict)

    return records, api_url


def main() -> None:
    _load_local_env()

    parser = argparse.ArgumentParser(
        description="Fetch FactSet consensus and segment consensus estimates into cutoff cache (postmortem only)."
    )
    parser.add_argument("--company-id", default="ECOR", help="Company runtime id (default: ECOR)")
    parser.add_argument("--ticker", required=True, help="Ticker symbol")
    parser.add_argument("--cutoff-date", required=True, help="Cutoff date YYYY-MM-DD")
    parser.add_argument(
        "--consensus-date",
        default=None,
        help="Perspective date for consensus estimates (default: earnings date or cutoff date). "
        "Use the earnings date to get the final street consensus before results.",
    )
    parser.add_argument(
        "--exchange-suffix",
        default=os.environ.get("FACTSET_EXCHANGE_SUFFIX", "US"),
        help="FactSet exchange suffix (default: US)",
    )
    parser.add_argument(
        "--lookback-quarters",
        type=int,
        default=20,
        help="Number of historical quarters to fetch (default: 20)",
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

    consensus_date = _parse_date(args.consensus_date) if args.consensus_date else cutoff
    if consensus_date is None:
        raise SystemExit(f"Invalid consensus date: {args.consensus_date}")

    import fds.sdk.FactSetEstimates

    company_id = args.company_id.upper()
    ticker = args.ticker.upper()
    ticker_fs = _factset_ticker(ticker, args.exchange_suffix)

    run_cache = _data_dir() / "companies" / company_id / "cache" / f"asof_{args.cutoff_date}"
    estimates_dir = run_cache / "estimates" / ticker
    estimates_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = estimates_dir / "fetch_manifest.json"

    if args.skip_if_present:
        existing = _parse_existing_manifest(manifest_path)
        if isinstance(existing, dict) and _has_existing_data(existing):
            print(f"Skipping FactSet estimates fetch; existing data found in {manifest_path}")
            return

    configuration = _factset_config()

    downloaded_estimates: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    print(f"Consensus perspective date: {consensus_date} (earnings date)")

    with fds.sdk.FactSetEstimates.ApiClient(configuration) as api_client:
        # 1. Fetch consensus estimates (as of earnings date for final street consensus)
        try:
            consensus, cons_url = _fetch_consensus(
                api_client, ticker_fs, consensus_date, args.lookback_quarters
            )
            cons_path = estimates_dir / f"consensus_asof_{args.cutoff_date}.json"
            cons_path.write_text(json.dumps(consensus, indent=2, default=str) + "\n")
            downloaded_estimates.append({
                "source_type": "estimate",
                "retrieval_method": "factset_estimates_api",
                "data_type": "consensus",
                "ticker": ticker,
                "factset_ticker": ticker_fs,
                "url": cons_url,
                "local_path": str(cons_path),
                "records_count": len(consensus),
                "published_date": args.cutoff_date,
            })
            print(f"Consensus estimates: {len(consensus)} records")
        except Exception as exc:
            errors.append({
                "step": "consensus_estimates",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"Error fetching consensus estimates: {exc}")

        # 1b. Fetch annual consensus estimates (CFY, NFY, NFY+1)
        try:
            annual_cons, annual_cons_url = _fetch_annual_consensus(
                api_client, ticker_fs, consensus_date
            )
            annual_cons_path = estimates_dir / f"consensus_annual_asof_{args.cutoff_date}.json"
            annual_cons_path.write_text(json.dumps(annual_cons, indent=2, default=str) + "\n")
            downloaded_estimates.append({
                "source_type": "estimate",
                "retrieval_method": "factset_estimates_api",
                "data_type": "consensus_annual",
                "ticker": ticker,
                "factset_ticker": ticker_fs,
                "url": annual_cons_url,
                "local_path": str(annual_cons_path),
                "records_count": len(annual_cons),
                "published_date": args.cutoff_date,
            })
            print(f"Annual consensus estimates: {len(annual_cons)} records")
        except Exception as exc:
            errors.append({
                "step": "consensus_annual",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"Error fetching annual consensus estimates: {exc}")

        # 2-3. Fetch quarterly segment consensus estimates (BUS + GEO)
        for seg_type in ("BUS", "GEO"):
            seg_label = seg_type.lower()

            try:
                seg_est, seg_est_url = _fetch_segment_estimates(
                    api_client, ticker_fs, consensus_date, args.lookback_quarters, seg_type
                )
                seg_est_path = estimates_dir / f"segment_estimates_{seg_label}.json"
                seg_est_path.write_text(json.dumps(seg_est, indent=2, default=str) + "\n")
                downloaded_estimates.append({
                    "source_type": "estimate",
                    "retrieval_method": "factset_estimates_api",
                    "data_type": f"segment_estimates_{seg_label}",
                    "ticker": ticker,
                    "factset_ticker": ticker_fs,
                    "url": seg_est_url,
                    "local_path": str(seg_est_path),
                    "records_count": len(seg_est),
                    "published_date": args.cutoff_date,
                })
                print(f"Segment estimates ({seg_type}): {len(seg_est)} records")
            except Exception as exc:
                errors.append({
                    "step": f"segment_estimates_{seg_label}",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"Error fetching segment estimates ({seg_type}): {exc}")

    manifest = {
        "created_at_utc": _now_iso(),
        "company_id": company_id,
        "ticker": ticker,
        "factset_ticker": ticker_fs,
        "cutoff_date": args.cutoff_date,
        "lookback_quarters": args.lookback_quarters,
        "downloaded_estimates_count": len(downloaded_estimates),
        "downloaded_estimates": downloaded_estimates,
        "errors": errors,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Downloaded estimate files: {len(downloaded_estimates)}")

    if not downloaded_estimates:
        raise SystemExit(
            "No FactSet estimates data was downloaded. "
            "Check credentials, ticker, and cutoff date."
        )


if __name__ == "__main__":
    main()
