#!/usr/bin/env python3
"""Fetch earnings call transcripts via FactSet Events and Transcripts SDK into cutoff-scoped cache."""

from __future__ import annotations

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
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
    import fds.sdk.EventsandTranscripts

    user_id = os.environ.get("FACTSET_USER_ID", "").strip()
    api_key = os.environ.get("FACTSET_API_KEY", "").strip()
    if not user_id or not api_key:
        raise SystemExit("Missing FACTSET_USER_ID or FACTSET_API_KEY in env or .env")
    return fds.sdk.EventsandTranscripts.Configuration(
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


def _has_existing_transcripts(manifest: dict[str, Any]) -> bool:
    downloaded = manifest.get("downloaded_transcripts")
    if not isinstance(downloaded, list):
        return False
    for item in downloaded:
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("local_path") or "")).expanduser()
        if path.exists() and path.is_file():
            return True
    return False


def _infer_quarter_from_headline(headline: str, event_date: Optional[date]) -> Optional[int]:
    """Try to extract quarter from the transcript headline like 'Q3 2024 Earnings Call'."""
    match = re.search(r"Q([1-4])", headline or "")
    if match:
        return int(match.group(1))
    if event_date:
        month = event_date.month
        # Earnings calls happen ~1 month after quarter end
        # Q1 (Jan-Mar) reported Apr-May, Q2 (Apr-Jun) reported Jul-Aug, etc.
        if month in (4, 5):
            return 1
        elif month in (7, 8):
            return 2
        elif month in (10, 11):
            return 3
        elif month in (1, 2):
            return 4
    return None


def _infer_fiscal_year_from_headline(headline: str, event_date: Optional[date]) -> Optional[int]:
    """Try to extract fiscal year from the transcript headline."""
    match = re.search(r"Q[1-4]\s+(\d{4})", headline or "")
    if match:
        return int(match.group(1))
    if event_date:
        quarter = _infer_quarter_from_headline(headline, event_date)
        if quarter == 4 and event_date.month in (1, 2):
            return event_date.year - 1
        return event_date.year
    return None


def _parse_xml_to_speakers(xml_content: str) -> list[dict[str, Any]]:
    """Parse FactSet CallStreet XML transcript into speaker segments.

    FactSet CallStreet XML structure:
      <TranscriptsCollection>
        <transcript>
          <meta>
            <participants>
              <participants id="0" type="operator"/>
              <participants id="1" type="corprep" affiliation="..." title="CEO"/>
            </participants>
          </meta>
          <body>
            <section name="MANAGEMENT DISCUSSION SECTION">
              <speaker id="1">
                <plist><p>...</p><p>...</p></plist>
              </speaker>
            </section>
          </body>
        </transcript>
      </TranscriptsCollection>

    Returns list of dicts with {name, title, text, section} matching the
    existing transcript JSON schema used by the agent.
    """
    speakers: list[dict[str, Any]] = []

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        text = re.sub(r"<[^>]+>", " ", xml_content)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            speakers.append({"name": "Unknown", "title": "Unknown", "text": text})
        return speakers

    # Build participant lookup: id -> {affiliation, title, type}
    participant_map: dict[str, dict[str, str]] = {}
    for p in root.iter("participants"):
        pid = p.get("id")
        if pid is not None and (p.get("type") or p.get("affiliation")):
            participant_map[pid] = {
                "affiliation": p.get("affiliation", ""),
                "title": p.get("title", ""),
                "type": p.get("type", ""),
                "entity": p.get("entity", ""),
            }

    # Parse body sections and speaker elements
    for section in root.iter("section"):
        section_name = section.get("name", "")
        for speaker_el in section.iter("speaker"):
            speaker_id = speaker_el.get("id", "")
            participant = participant_map.get(speaker_id, {})

            name = participant.get("affiliation", "Unknown")
            title = participant.get("title", "Unknown")
            ptype = participant.get("type", "")

            # Collect all <p> text within this speaker block
            paragraphs: list[str] = []
            for p_el in speaker_el.iter("p"):
                p_text = ET.tostring(p_el, encoding="unicode", method="text")
                p_text = re.sub(r"\s+", " ", p_text).strip()
                if p_text:
                    paragraphs.append(p_text)

            full_text = " ".join(paragraphs)
            if full_text:
                speakers.append({
                    "name": name,
                    "title": title,
                    "type": ptype,
                    "section": section_name,
                    "text": full_text,
                })

    # Fallback: extract all text as a single block
    if not speakers:
        all_text = ET.tostring(root, encoding="unicode", method="text")
        all_text = re.sub(r"\s+", " ", all_text).strip()
        if all_text:
            speakers.append({"name": "Unknown", "title": "Unknown", "text": all_text})

    return speakers


def _sanitize_filename(text: str, max_len: int = 80) -> str:
    """Sanitize a string for use as a filename."""
    text = re.sub(r"[^\w\s\-.]", "", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text[:max_len] if text else "slide"


def _fetch_investor_slides(
    api_client,
    ticker_fs: str,
    cutoff: date,
    max_results: int = 25,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch investor slide deck metadata from FactSet.

    Returns list of slide document dicts with slidesUrl, headline, storyDateTime.
    """
    from fds.sdk.EventsandTranscripts.api import transcripts_api

    api_instance = transcripts_api.TranscriptsApi(api_client)

    api_url = "https://api.factset.com/content/events/v2/transcripts/investor-slides"

    years_back = 3
    start_date = date(cutoff.year - years_back, 1, 1)

    response = api_instance.get_transcripts_investor_slides(
        ids=[ticker_fs],
        start_date=start_date,
        end_date=cutoff,
        sort=["-storyDateTime"],
        pagination_limit=max_results,
    )

    results: list[dict[str, Any]] = []
    if hasattr(response, "data") and response.data:
        for item in response.data:
            item_dict = item.to_dict() if hasattr(item, "to_dict") else {}
            docs = item_dict.get("documents") or []
            for doc in docs:
                if isinstance(doc, dict):
                    results.append(doc)

    return results, api_url


def _download_slide_pdf(
    slides_url: str,
    out_path: Path,
    user_id: str,
    api_key: str,
) -> bool:
    """Download a slide PDF from FactSet using basic auth. Returns True on success."""
    import requests

    try:
        resp = requests.get(slides_url, auth=(user_id, api_key), timeout=60)
        if resp.status_code == 200 and len(resp.content) > 100:
            out_path.write_bytes(resp.content)
            return True
        else:
            print(f"  Warning: slide download returned status {resp.status_code} "
                  f"({len(resp.content)} bytes) for {slides_url}")
    except Exception as exc:
        print(f"  Warning: slide download failed for {slides_url}: {exc}")
    return False


def _search_transcripts(
    api_client,
    ticker_fs: str,
    cutoff: date,
    max_periods: int,
) -> list[dict[str, Any]]:
    """Search for earnings call transcripts before the cutoff date.

    Returns flat list of document dicts (extracted from nested response).
    """
    from fds.sdk.EventsandTranscripts.api import transcripts_api
    from fds.sdk.EventsandTranscripts.models import (
        TranscriptsByIdsRequest,
        TranscriptsRequest,
        TranscriptsRequestMeta,
        TranscriptsRequestMetaPagination,
    )

    api_instance = transcripts_api.TranscriptsApi(api_client)

    # Search far enough back to get max_periods quarters
    years_back = (max_periods // 4) + 2
    start_date = date(cutoff.year - years_back, 1, 1)

    results: list[dict[str, Any]] = []
    offset = 0
    page_size = 25

    while True:
        request = TranscriptsRequest(
            data=TranscriptsByIdsRequest(
                ids=[ticker_fs],
                start_date=start_date,
                end_date=cutoff,
            ),
            meta=TranscriptsRequestMeta(
                pagination=TranscriptsRequestMetaPagination(
                    limit=page_size,
                    offset=offset,
                ),
                sort=["-storyDateTime"],
            ),
        )

        response = api_instance.search_transcripts(request)

        if not hasattr(response, "data") or not response.data:
            break

        page_docs: list[dict[str, Any]] = []
        for item in response.data:
            item_dict = item.to_dict() if hasattr(item, "to_dict") else {}
            # Response nests results inside "documents" array
            docs = item_dict.get("documents") or []
            for doc in docs:
                if isinstance(doc, dict):
                    page_docs.append(doc)

        results.extend(page_docs)

        if len(page_docs) < page_size:
            break
        offset += page_size

        if len(results) >= max_periods:
            break

    return results


def _fetch_transcript_content(
    api_client,
    report_id: str,
) -> Optional[str]:
    """Fetch full transcript content as XML.

    The SDK's get_transcriptsin_xml returns raw XML that its deserializer
    cannot parse (ApiTypeError: expects ResponseType, gets str).  Use
    _preload_content=False to get the raw urllib3 response and read the
    bytes directly.
    """
    from fds.sdk.EventsandTranscripts.api import transcripts_api

    api_instance = transcripts_api.TranscriptsApi(api_client)

    try:
        response = api_instance.get_transcriptsin_xml(
            report_ids=[report_id],
            format="ContentXML",
            _preload_content=False,
        )
        raw = response.read() if hasattr(response, "read") else response.data
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str) and raw.strip():
            return raw
    except Exception as exc:
        print(f"  Warning: could not fetch transcript content for report {report_id}: {exc}")

    return None


def main() -> None:
    _load_local_env()

    parser = argparse.ArgumentParser(
        description="Fetch FactSet earnings call transcripts into cutoff cache."
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
        "--max-periods",
        type=int,
        default=80,
        help="Maximum historical periods to fetch (default: 80)",
    )
    parser.add_argument(
        "--skip-if-present",
        action="store_true",
        help="Skip retrieval if manifest already has existing transcripts.",
    )
    args = parser.parse_args()

    cutoff = _parse_date(args.cutoff_date)
    if cutoff is None:
        raise SystemExit(f"Invalid cutoff date: {args.cutoff_date}")

    import fds.sdk.EventsandTranscripts

    company_id = args.company_id.upper()
    ticker = args.ticker.upper()
    ticker_fs = _factset_ticker(ticker, args.exchange_suffix)

    run_cache = _data_dir() / "companies" / company_id / "cache" / f"asof_{args.cutoff_date}"
    transcripts_dir = run_cache / "transcripts" / ticker
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = transcripts_dir / "fetch_manifest.json"

    if args.skip_if_present:
        existing = _parse_existing_manifest(manifest_path)
        if isinstance(existing, dict) and _has_existing_transcripts(existing):
            print(f"Skipping FactSet transcript fetch; existing transcripts found in {manifest_path}")
            return

    configuration = _factset_config()

    downloaded_transcripts: list[dict[str, Any]] = []
    unavailable_transcripts: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen_quarters: set[tuple[int, int]] = set()

    with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
        # 1. Search for transcripts
        print(f"Searching FactSet transcripts for {ticker_fs} before {args.cutoff_date}...")
        search_results = _search_transcripts(api_client, ticker_fs, cutoff, args.max_periods)
        print(f"Found {len(search_results)} transcript results")

        # 2. Fetch each transcript
        for result in search_results:
            report_id = str(result.get("report_id") or "").strip()
            event_date_raw = result.get("event_date")
            event_date = _parse_date(event_date_raw)
            headline = str(result.get("headline") or "")
            transcript_type = str(result.get("transcript_type") or "")
            transcripts_url = str(result.get("transcripts_url") or "")

            quarter = _infer_quarter_from_headline(headline, event_date)
            year = _infer_fiscal_year_from_headline(headline, event_date)

            if quarter is None or year is None:
                unavailable_transcripts.append({
                    "report_id": report_id,
                    "headline": headline,
                    "event_date": event_date.isoformat() if event_date else None,
                    "reason": "could_not_infer_quarter_or_year",
                })
                continue

            key = (year, quarter)
            if key in seen_quarters:
                continue
            seen_quarters.add(key)

            conference_date = event_date.isoformat() if event_date else None
            source_url = (
                f"https://api.factset.com/content/events/v2/transcripts"
                f"?reportId={report_id}"
            )
            out_path = transcripts_dir / f"Q{quarter}_{year}_l2.json"
            raw_xml_path = transcripts_dir / f"Q{quarter}_{year}_raw.xml"

            # Fetch full content via SDK (using _preload_content=False)
            xml_content = None
            if report_id:
                xml_content = _fetch_transcript_content(api_client, report_id)

            if xml_content is None:
                unavailable_transcripts.append({
                    "report_id": report_id,
                    "year": year,
                    "quarter": quarter,
                    "conference_date": conference_date,
                    "reason": "transcript_content_not_available",
                    "url": source_url,
                })
                continue

            # Save raw XML
            raw_xml_path.write_text(xml_content)

            # Parse XML to speaker-segmented JSON
            speakers = _parse_xml_to_speakers(xml_content)

            payload: dict[str, Any] = {
                "available": True,
                "ticker": ticker,
                "year": year,
                "quarter": quarter,
                "level": 2,
                "conference_date": conference_date,
                "transcript_type": transcript_type,
                "speakers": speakers,
            }
            out_path.write_text(json.dumps(payload, indent=2) + "\n")

            downloaded_transcripts.append({
                "source_type": "transcript",
                "retrieval_method": "factset_transcripts_api",
                "ticker": ticker,
                "factset_ticker": ticker_fs,
                "year": year,
                "quarter": quarter,
                "level": 2,
                "published_date": conference_date,
                "url": source_url,
                "local_path": str(out_path),
                "raw_xml_path": str(raw_xml_path),
                "speakers_count": len(speakers),
                "report_id": report_id,
                "transcript_type": transcript_type,
            })
            print(f"  Q{quarter}_{year}: {len(speakers)} speakers")

        # 3. Fetch investor slide deck metadata and download PDFs
        slides_dir = run_cache / "slides" / ticker
        slides_dir.mkdir(parents=True, exist_ok=True)
        downloaded_slides: list[dict[str, Any]] = []

        try:
            print(f"Searching FactSet investor slides for {ticker_fs}...")
            slide_docs, slides_api_url = _fetch_investor_slides(
                api_client, ticker_fs, cutoff
            )
            print(f"Found {len(slide_docs)} investor slide results")

            user_id = os.environ.get("FACTSET_USER_ID", "").strip()
            api_key = os.environ.get("FACTSET_API_KEY", "").strip()

            for doc in slide_docs:
                slides_url = doc.get("slides_url") or doc.get("slidesUrl") or ""
                headline = doc.get("headline") or "slides"
                story_dt = doc.get("story_date_time") or doc.get("storyDateTime")
                event_id = doc.get("event_id") or doc.get("eventId") or ""

                if not slides_url:
                    continue

                fname = _sanitize_filename(headline) + ".pdf"
                pdf_path = slides_dir / fname

                if _download_slide_pdf(slides_url, pdf_path, user_id, api_key):
                    downloaded_slides.append({
                        "source_type": "investor_slides",
                        "retrieval_method": "factset_transcripts_api",
                        "ticker": ticker,
                        "factset_ticker": ticker_fs,
                        "headline": headline,
                        "event_id": event_id,
                        "url": slides_url,
                        "local_path": str(pdf_path),
                        "published_date": str(story_dt)[:10] if story_dt else None,
                    })
                    print(f"  Downloaded slide: {fname}")
        except Exception as exc:
            errors.append({
                "step": "investor_slides",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"Error fetching investor slides: {exc}")

    manifest = {
        "created_at_utc": _now_iso(),
        "company_id": company_id,
        "ticker": ticker,
        "factset_ticker": ticker_fs,
        "cutoff_date": args.cutoff_date,
        "level": 2,
        "selection_limits": {
            "max_periods": max(1, args.max_periods),
        },
        "search_results_count": len(search_results),
        "downloaded_transcripts_count": len(downloaded_transcripts),
        "downloaded_transcripts": downloaded_transcripts,
        "downloaded_slides_count": len(downloaded_slides),
        "downloaded_slides": downloaded_slides,
        "unavailable_transcripts": unavailable_transcripts,
        "errors": errors,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote manifest: {manifest_path}")
    print(f"Downloaded transcripts: {len(downloaded_transcripts)}")

    if search_results and not downloaded_transcripts:
        raise SystemExit(
            "No transcripts were downloaded despite search results. "
            "Check FactSet credentials and transcript coverage."
        )
    if not search_results:
        raise SystemExit(
            "No transcript search results found. "
            "Check ticker and cutoff date."
        )


if __name__ == "__main__":
    main()
