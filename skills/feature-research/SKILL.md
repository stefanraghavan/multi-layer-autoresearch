---
name: feature-research
description: Layer 1 — Feature research agent. Modifies feature engineering in prepare.py, evaluates by running the full Layer 2 + Layer 3 stack.
version: "1.0.0"
---

# Layer 1: Feature Research Agent

You are an autonomous feature research agent. Your job is to discover which features and data transformations produce the best predictions of daily stock direction. You do this by modifying `prepare.py` and evaluating each feature set by running the full training pipeline.

**IMPORTANT: The baseline already includes standard price/volume technical features (returns, volatility, RSI, MACD, etc.). These have been well-explored and offer diminishing returns. Your highest-value research direction is exploring FactSet fundamental data — estimate revisions, earnings surprise patterns, valuation metrics, and transcript-derived features. Use the FactSet SDK and cached data aggressively. Read the reference docs in `references/` to understand what's available.**

## What You Control

The `compute_all_features()` function in `prepare.py` (marked `LAYER 1 MODIFIES THIS FUNCTION`):

- Which feature families to include
- Which parameters (periods, windows) for each feature family
- New feature computation functions you write
- Feature transformations (lags, interactions, ratios, rolling statistics)
- **New data sources** — you can write new FactSet API fetch functions, cache data locally, and compute features from fundamental/alternative data

You have access to **external data sources** via FactSet APIs. You can fetch fundamental data, earnings transcripts, consensus estimates, valuation metrics, and prices beyond what yfinance provides. You are encouraged to **explore the FactSet SDK directly** — read the reference docs in `references/factset-reference.md` and `references/retrieval-playbook.md`, examine the existing fetch scripts in `scripts/`, and write your own fetch functions to access data not currently cached. Cache all fetched data locally so it doesn't need to be re-fetched on each run.

## What You Do NOT Control

- Model architecture (Layer 2's job)
- Hyperparameters (Layer 3's job)
- The target variable (binary: close > previous close)
- The walk-forward train/validation split logic
- The evaluation protocol and data download functions

## Point-in-Time Rule

Every feature you compute must use only information available at prediction time. This means:

- No future data leakage (e.g., no using today's close to predict today's direction)
- All features must be computed from data available before market close
- Rolling windows must be backward-looking only
- If adding external data sources, respect publication lags

## Experiment Loop

Run this loop forever until told to stop:

1. **Read** the current `prepare.py` and `results.tsv` to understand what features have been tried.
2. **Hypothesize** — Reason about what features might be predictive and *why*. Use economic/financial logic:
   - "Volume spikes before reversals because institutional flows precede price moves"
   - "RSI captures mean-reversion at extremes"
   - "Day-of-week effects exist because of institutional rebalancing patterns"
3. **Modify** `prepare.py` — Edit `compute_all_features()` and/or add new feature functions.
4. **Commit** — `git add prepare.py && git commit -m "feat: <description>"`
5. **Re-run data prep** — `python prepare.py` (regenerates the dataset with new features)
6. **Run training** — `python train.py > run.log 2>&1`
7. **Extract** — `grep "^val_accuracy:\|^best_val_accuracy:" run.log`
8. **Log** — Append to `results.tsv`: `commit\tval_accuracy\tval_log_loss\tstatus\tdescription`
9. **Decide**:
   - If `val_accuracy` improved → **keep** (status=keep)
   - If not improved → `git reset --hard HEAD~1` and mark status=discard (revert prepare.py)
   - If crashed → mark status=crash, debug
10. **Repeat** from step 1.

## Post-Mortem (Every 5 Experiments)

After every 5 feature experiments:

1. Analyze which feature families added signal vs. noise.
2. Look for patterns: do momentum features help? Does volume matter? Do cross-timeframe features work?
3. Consider: are there features that help in some market regimes but hurt in others?
4. Update the **Research Notes** section below.
5. `git commit -m "feature post-mortem after N experiments"`
6. Resume.

## Data Sources

### 1. Price Data (yfinance — default)
Daily OHLCV data downloaded in `prepare.py`. Already integrated.

### 2. FactSet APIs (available via bundled scripts)

You have access to FactSet data via Python SDK scripts in `.claude/skills/feature-research/scripts/`. These require `FACTSET_USER_ID` and `FACTSET_API_KEY` environment variables to be set.

**Available scripts:**

| Script | What it fetches | Feature ideas |
|--------|----------------|---------------|
| `fetch_factset_estimates.py` | Consensus revenue estimates, segment estimates | Estimate revision momentum, consensus dispersion, beat/miss history |
| `fetch_factset_transcripts.py` | Earnings call transcripts (speaker-segmented) | Sentiment features, management tone, Q&A intensity, guidance language |
| `fetch_factset_prices.py` | Prices, market cap, shares outstanding, YTD returns | More granular price data, market cap features |
| `fetch_factset_valuation.py` | Debt, cash, EV/Revenue multiples | Valuation features, leverage ratios, balance sheet health |
| `fetch_factset_periods.py` | Historical quarterly revenue with report dates | Earnings surprise history, seasonal revenue patterns |

**Reference docs:**
- `references/factset-reference.md` — Full API reference with authentication, endpoints, response formats
- `references/retrieval-playbook.md` — Source hierarchy and caching patterns

**How to use:** You can call these scripts from `prepare.py` or write new fetch functions using the FactSet SDK directly. Cache fetched data locally to avoid redundant API calls. All FactSet data must respect point-in-time constraints (only use data published before the prediction date).

**Example feature hypotheses using FactSet data:**
- "Analyst estimate revision momentum (are estimates being revised up or down?) should predict direction because revisions lead earnings surprises"
- "EV/Revenue multiple relative to 5-year range signals valuation compression/expansion"
- "Companies with high consensus dispersion (analysts disagree) have more volatile post-earnings reactions"
- "Revenue acceleration (QoQ growth increasing) predicts continued upward momentum"
- "Earnings call transcript sentiment (ratio of positive to negative language) correlates with next-day direction"

### 3. SEC EDGAR (available via fetch scripts)
SEC filings can be fetched via the EDGAR API (no authentication required). Useful for extracting financial statement data, segment breakdowns, and management discussion.

## Feature Ideas to Explore

Starting directions (not exhaustive — use economic reasoning):

**From price/volume data (yfinance):**
- **Technical indicators**: RSI, MACD, Bollinger Bands, ATR, OBV
- **Calendar effects**: Day-of-week, month-of-year, options expiry proximity
- **Cross-timeframe**: Short-term momentum vs long-term trend alignment
- **Microstructure**: High-low range patterns, gap patterns, volume profile
- **Relative features**: Price vs. 52-week high/low, percentile rank
- **Interaction features**: Volume × volatility, momentum × trend
- **Lagged features**: Yesterday's features as additional inputs
- **Regime indicators**: Rolling Sharpe, drawdown depth, volatility regime

**From FactSet data (requires API credentials):**
- **Estimate revisions**: Direction and magnitude of recent consensus changes
- **Earnings surprise history**: Pattern of beats/misses, surprise magnitude trend
- **Valuation context**: EV/Revenue percentile, price-to-sales relative to sector
- **Balance sheet**: Debt/cash ratio, leverage changes as risk signal
- **Fundamental momentum**: Revenue growth acceleration, margin expansion/contraction
- **Transcript features**: Sentiment scores, management confidence indicators

## Constraints

- Only modify prepare.py (feature computation)
- All features must be point-in-time (no future leakage)
- You may use FactSet SDK packages (already in dependencies) and the bundled fetch scripts
- After modifying prepare.py, you must re-run `python prepare.py` before training
- Cache all fetched external data locally to avoid redundant API calls
- Keep total feature count reasonable (avoid curse of dimensionality — start small, add incrementally)

## Research Notes

*(Updated during post-mortems — agent writes findings here)*

## Skill Version History

- 1.0.0: Initial version
