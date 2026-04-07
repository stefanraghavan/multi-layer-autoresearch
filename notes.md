# Multi-Layer Autoresearch — Final Research Summary

## Project Overview

Built and tested an autonomous feature research system inspired by the autoresearch paradigm. An LLM agent (GPT-5.4) autonomously proposes features, fetches data from FactSet APIs, tests each feature against a locked model, keeps improvements, discards regressions, and writes postmortem analyses every 10 experiments to guide future research.

**Total: 215+ experiments, ~$500-700 API cost, ~40+ hours runtime across 5 runs**

---

## Architecture Evolution

| Phase | Architecture | Outcome |
|-------|-------------|---------|
| 1 | Three layers (features → architecture → hyperparameters) | Architecture search found simplicity wins; hyperparameters exhausted in 20 experiments |
| 2 | Two layers (features → hyperparameters) | Hyperparameters converged to same optimum every time |
| 3 | One layer (features only) with MLP | Feature research produced genuine insights but MLP couldn't extract all signal |
| 4 | One layer with XGBoost | Immediately beat MLP by 2.6pp on same features |
| Final | XGBoost + confidence thresholds + walk-forward validation | Revealed regime-dependent signal at 56-63% on high-conviction predictions |

**Key learning:** For tabular financial prediction, the model and hyperparameters matter much less than what data you feed in. The autoresearch loop is most valuable as a feature research tool.

---

## Run-by-Run Results

### Run 1: Raw Daily Direction with MLP
- **Target:** Binary up/down across 30 stocks
- **45 experiments, ~5 hours**
- **Baseline 52.7% → Best 53.0%** (+0.3pp)
- Ceiling is market's up-day bias. Cash-flow fundamentals only marginal help.

### Run 2: Sector-Relative Direction with MLP
- **Target:** Did stock outperform its sector today? (30 stocks, all days)
- **66 experiments, ~13 hours**
- **Baseline 51.5% → Best 52.2%** (+0.7pp)
- Transcript tone features worked for sector-relative (failed for raw direction).
- Dropping noisy features improved accuracy.

### Run 3: Event-Window Sector-Relative with MLP
- **Target:** Sector-relative, only within 10 days of earnings (30 stocks)
- **81 experiments, ~15 hours**
- **Baseline 52.3% → Best 54.0%** (+1.7pp)
- EPS surprise features strongest. Feature space exhausted after 30 experiments.

### Run 4: Event-Window Sector-Relative with XGBoost
- **Target:** Same as Run 3 but with XGBoost
- **23 experiments, ~3 hours**
- **Baseline 53.6% → Best 54.5%** (+0.9pp)
- XGBoost baseline already higher than MLP's best-ever.
- Removing stale long-horizon features (20d return, 60d vol, 200d MA) helped.

### Run 5: Confidence Thresholds + Walk-Forward Validation
- **100 stocks, 6 annual walk-forward windows, 3 seeds each**
- **Overall at 0.55 threshold: 55.9% avg hit rate, ~35 trades/year**
- **Best regime (2024): 62.7% hit rate on 83 trades**
- **Worst regimes (2022-2023): near-zero confident predictions**
- Sector one-hot encoding tested and reverted (made results worse)

---

## Key Research Findings

### 1. Cash-flow fundamentals predict stock-specific alpha
FCF yield, operating CF yield, operating margin changes, and operating leverage deltas are the most robust FactSet-derived features across all runs. Accounting ratios (ROA, asset turnover, accruals) and balance sheet metrics (working capital, debt ratios) consistently underperformed.

### 2. Transcript tone predicts sector-relative outperformance
Management sentiment, Q&A intensity (analyst_share × log word count), and guidance confidence add signal for sector-relative prediction but not for raw direction. The key design choice: multiply all transcript features by a 30-day exponential recency decay so they're strongest right after earnings and fade to zero. Filtering to same/next-day transcript availability improved quality further.

### 3. Feature pruning is as valuable as feature addition
Removing stale/noisy features consistently improved accuracy:
- Dropping long-horizon technicals (20d return, 60d vol, 200d MA) in event windows
- Dropping consensus revisions and analyst coverage features
- The agent's postmortem: "slow macro/trend state is often stale relative to the current event"

### 4. Event-window filtering concentrates signal
Restricting to 10 days post-earnings where fundamental features are fresh improved accuracy by ~2pp vs predicting every day. 80% of training data was on days where FactSet features were stale — removing that noise helped.

### 5. XGBoost extracts more signal than MLP from the same features
XGBoost's baseline (53.6%) exceeded MLP's best after 81 experiments (54.0%). Gradient boosted trees are the right model for tabular financial data at this scale (~25K samples, ~40 features).

### 6. The signal is regime-dependent
Walk-forward validation across 6 annual windows showed the model finds clear patterns in some years (2024: 63% hit rate) and correctly abstains in others (2022-2023: near-zero trades). This is realistic behavior — post-earnings patterns are clearer in some market environments than others.

### 7. Sector encoding doesn't help
Adding sector one-hot features made results worse. XGBoost already learns sector-relevant patterns implicitly through the FactSet features (estimate revisions naturally vary by sector).

### 8. The autoresearch loop works for financial feature research
The agent autonomously:
- Called the FactSet SDK to fetch fundamentals, transcripts, and estimates
- Wrote 300+ lines of new data pipeline code
- Designed economically-motivated features (recency-weighted transcript tone, operating leverage deltas)
- Learned to remove features, not just add them
- Produced genuinely insightful postmortem analyses
- Discovered that transcript sentiment predicts sector-relative alpha — a non-obvious finding

---

## What Didn't Work

- **Transcript sentiment for raw direction prediction** — tone is stock-specific, not market-wide
- **Accounting ratios across sectors** — ROA, margins, turnover too heterogeneous when pooled
- **Daily mark-to-market valuation overlays** — reintroduce price noise the model already has
- **Consensus estimate revisions** — too noisy/sparse for this target
- **Accrual quality, working capital, balance sheet stress** — fragile across sectors
- **Post-report drift interactions** — model absorbs info without hand-crafted multiplications
- **Share dilution/buyback** — quarter-to-quarter changes too small and noisy
- **Sector one-hot encoding** — confused XGBoost with sparse features

---

## System Architecture (Final)

```
Feature Research Agent (GPT-5.4)
  ├── Reads prepare.py, results.tsv, research notes
  ├── Proposes one feature change per experiment
  ├── Can fetch new FactSet data via SDK
  ├── Can add, remove, or modify features
  ├── 3-run averaging (different seeds) per experiment
  ├── Keep if val_accuracy > best, else git revert
  ├── Postmortem every 10 experiments
  └── Accumulates knowledge across sessions

Model: XGBoost (500 estimators, depth 4, early stopping 50)
Target: Binary sector-relative direction (event-window only)
Data: 100 S&P 500 stocks, 8yr train / 2yr val, 10-day post-earnings windows
Evaluation: Walk-forward across 6 annual windows, confidence threshold filtering
```

---

## Honest Assessment

**Is this tradeable?** Not as a standalone strategy. 56% hit rate on ~35 trades/year with regime dependency is a thin edge. After transaction costs and the overhead of maintaining FactSet data feeds and model infrastructure, the expected return is marginal.

**What is it good for?**
1. **Research tool** — the autoresearch system produces genuine insights about what financial data predicts stock-specific outcomes. The transcript finding alone is publishable research.
2. **Signal component** — the features and confidence thresholds could be one input into a broader multi-signal trading system alongside other alpha sources.
3. **Methodology proof-of-concept** — autonomous LLM-driven feature research with postmortem loops works and produces results a human quant researcher would need weeks to replicate.

---

## Potential Future Directions

1. **Predict magnitude, not direction** — regression on outperformance magnitude with position sizing by predicted alpha
2. **Sector-specific models** — train separate XGBoost per sector where data permits (tech: 20 stocks, healthcare: 15)
3. **Combine with hedge fund system** — use the earnings projection system's revenue surprise as a feature (it has 75% training hit rate on surprise direction)
4. **Longer event windows** — 20-30 days might capture post-earnings drift more fully
5. **Rolling retraining** — retrain monthly instead of annually so features stay fresh
6. **Live paper trading** — deploy the confidence-threshold strategy on paper to validate in real time
7. **More alternative data** — satellite, credit card, web traffic via other APIs

---

## Files

| File | Purpose |
|------|---------|
| `orchestrator.py` | Runs the feature research loop |
| `prepare.py` | Data prep: download, features, event filtering, sector-relative target |
| `train.py` | XGBoost training with seed support |
| `fetch_factset.py` | Cache FactSet consensus estimates |
| `analyze_confidence.py` | Confidence threshold analysis |
| `analyze_walkforward.py` | Rolling walk-forward validation |
| `lib/agent.py` | OpenAI Responses API client |
| `lib/phases.py` | Feature experiment prompt and result parsing |
| `lib/config.py` | Configuration |
| `lib/db.py` | SQLite experiment tracking |
| `skills/feature-research/SKILL.md` | Agent instructions + accumulated research notes |
| `data/` | FactSet caches (consensus, fundamentals, transcripts) |
