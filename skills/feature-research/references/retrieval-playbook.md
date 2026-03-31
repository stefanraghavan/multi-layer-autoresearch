# Retrieval Playbook

Use this guide when you need to fetch data for projection runs.

In `NETWORK_POLICY=api_only`:
The agent handles all API retrieval directly via Bash scripts, enforcing allowlist, cutoff, and provenance rules.

## Source Hierarchy
1. SEC EDGAR filings (10-K, 10-Q, 8-K) are the **primary source of truth** for historical financials. Raw filings contain segment detail tables, channel-level revenue breakdowns, MD&A narrative, and risk factors at the granularity management actually reports. **Read the raw filings** to build your historical segment model.
2. FactSet Consensus Estimates — `consensus_m` (total revenue) is provided as a prompt input. Detailed estimate files (segment consensus, guidance) are available at `cache/asof_{cutoff}/estimates/{TICKER}/` for comparison after you build your bottom-up projection.
3. FactSet earnings transcripts (`fds.sdk.EventsandTranscripts`) provide qualitative context (drivers, headwinds, one-time items, guidance commentary).
4. FactSet investor slides and conference presentations — attached as PDF document inputs when available. Date-filtered from FactSet (safe for backtesting). Check for attached slide PDFs for TAM data, product roadmaps, and visual segment breakdowns.
5. Proprietary research reports (e.g., Evercore ISI) — attached as PDF document inputs when available. Fetched from research portals with cutoff date enforcement. Use for sell-side consensus context, segment estimates, and channel checks.

Sources 4 and 5 should never override SEC/FactSet data, but provide valuable supplemental context when available.

## Cutoff and Provenance Rules
- Only use sources with `published_date <= cutoff_date`.
- Do not use filing/press-release/slide sources published after `target_period` end.
- Every external source used must be listed in `sources_used` with:
  - `type`
  - `url`
  - `published_date` (if available)
  - `local_path`
  - `retrieval_method` (`sec_edgar_api`, `factset_transcripts_api`, `factset_estimates_api`, `browser_use_api`, etc.)
- Save artifacts under `cache/asof_{cutoff_date}/...`.

## Cache Locations

### Projection Sources (available during projection)
- SEC EDGAR filings: `cache/asof_{cutoff_date}/filings/{TICKER}/{CIK}/*.html` (10-K, 10-Q, 8-K)
- Transcripts: `cache/asof_{cutoff_date}/transcripts/{TICKER}/...`
- Slides: `cache/asof_{cutoff_date}/slides/{TICKER}/...`
- Press releases: `cache/asof_{cutoff_date}/press_releases/{TICKER}/...`
- Proprietary reports: `cache/asof_{cutoff_date}/proprietary_reports/{provider}/{TICKER}/...`

### FactSet Estimates (available for comparison)
- Consensus: `cache/asof_{cutoff_date}/estimates/{TICKER}/consensus_asof_{cutoff_date}.json`
- Segment estimates (BUS, quarterly): `cache/asof_{cutoff_date}/estimates/{TICKER}/segment_estimates_bus.json`
- Segment estimates (GEO, quarterly): `cache/asof_{cutoff_date}/estimates/{TICKER}/segment_estimates_geo.json`

**Build your bottom-up projection first**, then use these for comparison and sense-checking.

## API & Tool References
- FactSet: [factset-reference.md](factset-reference.md)
- SEC EDGAR: [sec-edgar-reference.md](sec-edgar-reference.md)
- Browser Use HTTP API (IR): `scripts/fetch_ir_artifacts.py` (see `script-starters.md` Pattern C)
- Browser Use HTTP API (proprietary): `scripts/fetch_proprietary_reports.py` (see `script-starters.md` Pattern C1)

## Mandatory Retrieval Sequence

**You must fetch all data sources below before starting analysis.** Check cache first — if the file exists, read it. If not, run the bundled script (or equivalent) to fetch it. An empty cache is not an excuse to skip a source.

### Step 1: SEC EDGAR Filings (required — primary data source)
Run `scripts/fetch_sec_filings.py` to download raw 10-K, 10-Q, and 8-K filings. These are the **primary source** for:
- **Quarterly and annual revenue** (from financial statements)
- **Segment detail tables** with channel-level revenue breakdowns (VA/DoD, commercial, NHS, etc.)
- **8-K earnings press releases** with same-day revenue breakdowns — often more granular than 10-Q filings (e.g., channel-level splits like VA/DoD vs commercial that 10-Qs only report as geographic aggregates)
- **MD&A narrative** with management's discussion of drivers, headwinds, and outlook
- **Risk factors** that may affect projections

The script respects the cutoff date — only filings with `filing_date <= cutoff_date` are downloaded.

**Read the downloaded HTML filings** to extract segment/channel breakdowns going back as far as available. This is critical for building a deep historical segment model. SEC filings contain the actual reported segment tables at the granularity management uses — far more detailed than any structured API. **Prioritize 8-K press releases** for sub-segment detail — they are filed on earnings day and often contain the most granular revenue breakdown available.

Also use the SEC EDGAR XBRL Company Facts API (see `sec-edgar-reference.md`) to pull quarterly revenue totals programmatically for trend analysis.

### Step 2: FactSet Transcripts (required)
Run `scripts/fetch_factset_transcripts.py` (or equivalent) to populate:
- `transcripts/{TICKER}/Q{q}_{year}_l2.json` (speaker-segmented JSON)
- `slides/{TICKER}/*.pdf` (investor slides, if entitled)

### Step 3: IR Slides and Press Releases (SKIPPED for backtesting)
**Do NOT run `scripts/fetch_ir_artifacts.py` during backtesting.** Browser Use scrapes the live IR website, which always serves the current corporate presentation — not the historical version. This creates data leakage (e.g., a 2026 slide deck containing FY 2022-2024 actual revenue downloaded into a 2023-cutoff cache). SEC EDGAR 8-K filings already provide historical earnings press releases with correct dates.

### Step 4: Proprietary Reports (required attempt)
Run `scripts/fetch_proprietary_reports.py` when credentials are available. Log the attempt even if no reports are found.

You may adapt or rewrite these scripts as needed — the data coverage is mandatory, not the specific script implementation.

## Transcript Retrieval Details
The bundled `scripts/fetch_factset_transcripts.py` handles the full workflow:
1. Search transcripts via FactSet `TranscriptsApi.search_transcripts()` with `end_date=cutoff_date` and `event_type="Earnings"`.
2. Fetch full transcript content via `get_transcriptsin_xml()` (ContentXML format).
3. Parse XML to speaker-segmented JSON; save in cutoff cache.
4. Attempt investor slide download via `get_transcripts_investor_slides()`.

If you write custom transcript retrieval, ensure the same output format. Use transcript findings to normalize and interpret trends; do not override SEC EDGAR reported data.

## Investor Slides Retrieval Workflow
The transcript script attempts to fetch investor slide deck metadata via `TranscriptsApi.get_transcripts_investor_slides()` and downloads PDFs using FactSet basic auth. Slides are saved to `cache/asof_{cutoff_date}/slides/{TICKER}/`.

This endpoint requires a separate FactSet entitlement and may return 403 if not subscribed. If FactSet returns no slide results or an auth error, fall back to Browser Use IR retrieval below.

## IR Retrieval Workflow (DISABLED for Backtesting)
**Do NOT use Browser Use for IR slide deck retrieval during backtesting.** Live IR sites always serve the current corporate presentation, not historical versions. This creates data leakage — e.g., a January 2026 slide deck containing FY 2022-2024 actual revenue was downloaded into a 2023-cutoff cache, contaminating the projection context.

SEC EDGAR 8-K filings already provide historical earnings press releases with correct filing dates. Use those instead for historical press release data.

## Proprietary Retrieval Workflow (Required Attempt)
**Always attempt** portal login and search when credentials are available. Log the outcome even if no documents are found for this company.

Provider baseline:
- Default provider is Evercore ISI portal (`evercoreisi.mediasterling.com`) when credentials are available.
- Keep retrieval company-agnostic: parameterize by `company_id`, `company_name`, `ticker`, and `cutoff_date`.

Credentials (from environment variables):
- `PROPRIETARY_RESEARCH_PORTAL_URL` — portal URL
- `PROPRIETARY_RESEARCH_USERNAME` — login username
- `PROPRIETARY_RESEARCH_PASSWORD` — login password
- `BROWSER_USE_API_KEY` — required for Browser Use API

Workflow:
1. Run the bundled `scripts/fetch_proprietary_reports.py` with company/ticker/cutoff parameters.
   See `script-starters.md` Pattern C1 for the full command template.
2. The script uses the Browser Use HTTP API (not CLI) to create a session, dispatch an agent to
   authenticate and search, then download PDFs via a 3-tier strategy:
   - Primary task `outputFiles` attachment
   - Secondary download task in the same API session
   - Direct `requests.Session` fallback with ASP.NET forms auth
3. The script enforces cutoff filtering and writes a `fetch_manifest.json` with full provenance.
4. Read the manifest to extract `downloaded_files`, `task_id`, and report metadata for trace.
5. Save artifacts and record provenance including `browser_use_task_id` from the manifest.
6. If no reports are downloadable, the manifest still records the attempt (query used, reason no files were downloaded). Include this in trace.

## Important
- Treat SEC EDGAR filings as authoritative for reported financials and segment breakdowns.
- `consensus_m` (total revenue) is provided as a prompt input. Detailed FactSet estimate files are available for comparison — but **build your bottom-up projection first** from SEC filings and transcripts before comparing to consensus.
- Use proprietary/IR content to refine assumptions, not to rewrite reported history.
