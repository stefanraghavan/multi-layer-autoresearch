# FactSet API Reference

## Authentication

All FactSet SDKs use the same authentication pattern:
```python
configuration = fds.sdk.<Package>.Configuration(
    username=os.environ["FACTSET_USER_ID"],   # e.g. "Formula_Ins-2313321"
    password=os.environ["FACTSET_API_KEY"],
)
```

## Ticker Format

FactSet uses exchange-qualified tickers: `"ECOR-US"` for NASDAQ/NYSE-listed US equities.
Convert plain tickers: `f"{ticker}-{suffix}"` where suffix defaults to `"US"`.

## Rate Limits

All APIs: **10 requests/second, 10 concurrent requests per user**.

---

## 1. Events and Transcripts (`fds.sdk.EventsandTranscripts`)

Base URL: `https://api.factset.com/content/events/v2`

### Search Transcripts

```python
from fds.sdk.EventsandTranscripts.api import transcripts_api
from fds.sdk.EventsandTranscripts.models import *  # includes TranscriptsByIdsRequest

request = TranscriptsRequest(
    data=TranscriptsByIdsRequest(      # NOT TranscriptsRequestData (discriminated union)
        ids=["ECOR-US"],
        start_date=date(2022, 1, 1),
        end_date=date(2024, 5, 1),       # cutoff date
    ),
    meta=TranscriptsRequestMeta(
        pagination=TranscriptsRequestMetaPagination(limit=25, offset=0),
        sort=["-storyDateTime"],
    ),
)
response = api_instance.search_transcripts(request)
# Results are nested: response.data[0].documents[] contains document dicts
```

**Response fields**: `report_id`, `event_date`, `headline`, `transcript_type`, `transcripts_url`

### Fetch Full Transcript

```python
response = api_instance.get_transcriptsin_xml(
    report_ids=["3022837"],
    format="ContentXML",    # XML | ContentXML | PDF | DocViewer
)
```

**Transcript types**: Corrected (edited, 1-3 days), Raw (hours), NearRealTime (during call)

**Event types**: `"Earnings"`, `"Guidance"`, `"ConferencePresentation"`, `"SalesRevenue"`

---

## 2. FactSet Estimates (`fds.sdk.FactSetEstimates`)

Base URL: `https://api.factset.com/content/factset-estimates/v2`

**Phase availability**: FactSet Estimates endpoints are used only during postmortem (consensus and segment consensus for error attribution). During projection, the agent does not access FactSet Estimates — it uses SEC EDGAR for historical financials and transcripts for guidance.

### Consensus Estimates (PIT) — Postmortem Only

```python
from fds.sdk.FactSetEstimates.api import consensus_api

response = api_instance.get_fixed_consensus(
    ids=["ECOR-US"],
    metrics=["SALES"],
    start_date=date(2024, 5, 1),      # perspective date (PIT)
    end_date=date(2024, 5, 1),
    frequency="D",
    fiscal_period_start="2024/1F",     # Q1 2024
    fiscal_period_end="2024/4F",       # Q4 2024
    periodicity="QTR",
    currency="USD",
)
```

**Fiscal period format**: `"2024/1F"` (Q1), `"2024/2F"` (Q2), `"2024/3F"` (Q3), `"2024/4F"` (Q4), `"2024"` (annual)

**Response fields**: `mean`, `median`, `high`, `low`, `num_estimates`, `fiscal_year`, `fiscal_period`

### Segment Consensus Estimates (Quarterly) — Postmortem Only

```python
from fds.sdk.FactSetEstimates.api import segments_api

response = api_instance.get_segments(
    ids=["ECOR-US"],
    metrics=["SALES"],
    segment_type="GEO",       # BUS or GEO
    periodicity="QTR",
    start_date=date(2023, 3, 8),     # consensus perspective date
    end_date=date(2023, 3, 8),
    frequency="D",
    relative_fiscal_start=-12,
    relative_fiscal_end=0,
    currency="USD",
)
```

**Response fields**: `segment_label`, `mean`, `median`, `high`, `low`, `estimate_count`, `fiscal_period`, `fiscal_year`, `fiscal_end_date`, `estimate_date`

### Investor Slides

```python
from fds.sdk.EventsandTranscripts.api import transcripts_api

response = api_instance.get_transcripts_investor_slides(
    ids=["ECOR-US"],
    start_date=date(2020, 1, 1),
    end_date=date(2023, 3, 2),   # cutoff date
    sort=["-storyDateTime"],
    pagination_limit=25,
)
# Returns metadata; download PDFs from slidesUrl with authenticated requests.get()
```

**Response fields**: `slidesUrl` (PDF download URL), `headline`, `storyDateTime`, `eventId`, `primaryIds`, `categories`

**Note**: Returns metadata only — PDF content must be downloaded separately via `slidesUrl` using FactSet basic auth credentials. Requires separate FactSet entitlement; returns 403 if not subscribed. Falls back to Browser Use IR retrieval when unavailable.

---

## PIT for Backtesting (Summary)

| Data Type | PIT Mechanism |
|---|---|
| SEC EDGAR filings | Filter `filing_date <= cutoff_date` in fetch script |
| Transcripts | Filter `end_date` = cutoff date (only published before cutoff) |
| Consensus *(postmortem only)* | `start_date` = `end_date` = cutoff date (consensus as-of snapshot) |

---

## Caching Patterns

| Data | Cache Path |
|---|---|
| SEC EDGAR filings (primary) | `cache/asof_{cutoff}/filings/{TICKER}/{CIK}/*.html` |
| Transcripts | `cache/asof_{cutoff}/transcripts/{TICKER}/Q{q}_{year}_l2.json` |
| Raw XML | `cache/asof_{cutoff}/transcripts/{TICKER}/Q{q}_{year}_raw.xml` |
| Consensus *(postmortem only)* | `cache/asof_{cutoff}/estimates/{TICKER}/consensus_asof_{cutoff}.json` |
| Segment Estimates (BUS) *(postmortem only)* | `cache/asof_{cutoff}/estimates/{TICKER}/segment_estimates_bus.json` |
| Segment Estimates (GEO) *(postmortem only)* | `cache/asof_{cutoff}/estimates/{TICKER}/segment_estimates_geo.json` |
| Investor Slides | `cache/asof_{cutoff}/slides/{TICKER}/*.pdf` |

---

## Bundled Scripts

| Script | Purpose |
|---|---|
| `scripts/fetch_sec_filings.py` | SEC EDGAR 10-K/10-Q/8-K filings (primary data source) |
| `scripts/fetch_factset_transcripts.py` | Earnings call transcripts (XML → JSON) + investor slide PDFs |
