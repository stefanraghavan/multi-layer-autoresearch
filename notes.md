# Multi-Layer Autoresearch — Research Notes

## Run 1: Raw Daily Direction (completed)

**45 experiments, ~5 hours**

Predicted raw daily up/down direction across 30 stocks. Hit ~53% accuracy ceiling (baseline ~52.7% up-day bias).

**What worked:** Cash-flow fundamentals (FCF yield, operating CF yield, operating leverage deltas).

**What didn't work:** Transcript sentiment, accounting ratios, consensus dispersion, daily valuation overlays, accrual quality, balance sheet stress, segment concentration.

**Key insight:** "The most robust signal comes from PIT-safe, low-frequency, economically interpretable fundamentals tied to realized operating performance — especially cash generation and operating-leverage changes."

**Conclusion:** The ~53% ceiling is likely the market's up-day bias (~52.7%) plus minimal alpha. Raw direction prediction on liquid large-caps is too dominated by market beta.

---

## Run 2: Sector-Relative Direction (completed)

**66 experiments, ~13 hours**

Switched target to sector-relative: "did the stock outperform its sector today?" Strips out market/sector beta and focuses on stock-specific alpha.

- **Baseline:** 51.5% (close to 50% by construction)
- **Best:** 52.2% (+0.75pp improvement)
- **15 keeps, 51 discards**

### What worked (sector-relative)

1. **Transcript tone features with recency decay** — management sentiment, sentiment spread (mgmt vs analyst), guidance confidence, all multiplied by 30d exponential decay. These *failed* for raw direction but *worked* for sector-relative. Key insight: management tone predicts stock-specific alpha, not market direction.

2. **Filtering transcript freshness** — restricting to same/next-day availability relative to conference date improved signal. Stale transcripts add noise.

3. **Dropping noisy features** — removing consensus revisions and analyst coverage features *improved* accuracy. Less is more.

4. **Repaired earnings surprise features** — fixed broken all-zero surprise features into PIT 30d recency-decayed event signals.

5. **Operating margin QoQ/YoY changes** — recency-weighted profitability change events from quarterly reports.

6. **FCF margin change features** — quarterly free-cash-flow margin changes as event signals.

### What didn't work (sector-relative)

- Share dilution/buyback proxies
- Accrual quality
- Asset turnover
- EV/TTM-sales relative valuation
- Capex intensity
- Debt-to-assets changes
- Report-date sales growth events
- Sector-relative event features
- Various transcript metadata pruning attempts
- Most "refinements" of already-working features

### Agent behavior observations

- The agent autonomously called FactSet SDK to fetch fundamentals, profitability data, and transcript data
- It wrote 300+ lines of new fetch/cache/compute code in prepare.py
- It learned to *remove* features, not just add — dropping consensus revisions was a key improvement
- The 3-run averaging (different seeds) effectively filtered noise
- Postmortems every 10 experiments produced genuinely insightful research notes
- After ~40 experiments, hit diminishing returns — last 25 experiments were all discards

### Key finding: transcript sentiment predicts sector-relative alpha

The most interesting discovery: transcript tone features that were useless for raw direction prediction became the strongest signal for sector-relative prediction. Management tone, guidance confidence, and the spread between management and analyst sentiment predict whether a stock outperforms its sector in the days following an earnings call. This makes economic sense — management tone is stock-specific information that doesn't move the market, but does differentiate individual stocks from their peers.

---

## Setup (current)

- **Model**: Simple MLP [64, 32] with LayerNorm, GELU, Dropout — locked
- **Hyperparameters**: Locked (AdamW, LR=1e-3, batch=256, dropout=0.2, constant schedule)
- **Data**: 30 liquid stocks across 6 sectors, 8 years training / 2 years validation
- **Evaluation**: 3-run averaged val_accuracy, single walk-forward split
- **Agent**: GPT-5.4 via OpenAI Responses API
- **FactSet access**: Estimates, Fundamentals, Transcripts, Prices SDKs

---

## Next steps to explore

### 1. Rolling walk-forward validation
Current setup uses a single train/val split. Rolling walk-forward (multiple windows across different market regimes) would give more confidence in results and catch regime-dependent overfitting.

### 2. Ticker/sector encoding
The model doesn't know which stock it's looking at. Adding a categorical feature for sector would let it learn sector-specific patterns (e.g., transcripts matter more for tech than utilities).

### 3. Single sector focus
Train only on 10 tech stocks where FactSet features should be most homogeneous. Reduces noise from cross-sector heterogeneity (FCF margin means different things for JPM vs NVDA).

### 4. Gradient boosted trees
XGBoost/LightGBM consistently beats neural networks on tabular data this size. The features we discovered (transcript tone, operating margin changes, surprise PEAD) could perform better with a GBT.

### 5. Event-driven subsystem
Instead of predicting every day, only predict on days near earnings announcements where transcript/surprise features are freshest. The hedge fund system showed 75% hit rate on earnings-driven signals.

### 6. Conditional prediction with abstention
Predict with a confidence threshold — only trade when the model's probability is above 55% or below 45%. Could dramatically improve hit rate at the cost of fewer predictions.

### 7. Multi-horizon targets
The ML engineer recommended predicting 1/5/20-day forward excess returns with multi-task heads. More aligned with how quant funds actually trade.

---

## Architecture evolution history

1. **Three layers**: Feature research → Architecture search → Hyperparameter tuning (original design)
2. **Two layers**: Feature research → Hyperparameter tuning (architecture locked — simplicity won)
3. **One layer**: Feature research only (hyperparameters locked — search exhausted in ~20 experiments)

This progression validated that for tabular financial prediction, the model and hyperparameters matter much less than what data you feed in. The autoresearch loop is most valuable as a **feature research tool** — systematically exploring data sources, testing hypotheses with economic reasoning, and accumulating knowledge through postmortems.

## Total experiments run: ~110+
## Total cost: ~$200-400 (OpenAI API)
## Total runtime: ~20+ hours
