# Multi-Layer Autoresearch — Research Notes

## Run 1: Raw Daily Direction with MLP (completed)

**45 experiments, ~5 hours**

Predicted raw daily up/down direction across 30 stocks. Hit ~53% accuracy ceiling (baseline ~52.7% up-day bias).

**What worked:** Cash-flow fundamentals (FCF yield, operating CF yield, operating leverage deltas).

**What didn't work:** Transcript sentiment, accounting ratios, consensus dispersion, daily valuation overlays, accrual quality, balance sheet stress, segment concentration.

**Key insight:** "The most robust signal comes from PIT-safe, low-frequency, economically interpretable fundamentals tied to realized operating performance — especially cash generation and operating-leverage changes."

**Conclusion:** The ~53% ceiling is likely the market's up-day bias (~52.7%) plus minimal alpha. Raw direction prediction on liquid large-caps is too dominated by market beta.

---

## Run 2: Sector-Relative Direction with MLP (completed)

**66 experiments, ~13 hours**

Switched target to sector-relative: "did the stock outperform its sector today?" Strips out market/sector beta.

- **Baseline:** 51.5% → **Best: 52.2%** (+0.75pp)
- Transcript tone features worked for sector-relative (failed for raw direction)
- Dropping noisy features (consensus revisions) improved accuracy
- Feature research exhausted after ~40 experiments

---

## Run 3: Event-Window Sector-Relative with MLP (completed)

**81 experiments, ~15 hours**

Filtered to only days within 10 days of earnings announcements. Every sample has fresh transcript/surprise data.

- **Baseline:** 52.3% → **Best: 54.0%** (+1.7pp)
- EPS surprise features were the strongest single addition
- Feature space exhausted after ~30 experiments, 50+ consecutive discards

---

## Run 4: Event-Window Sector-Relative with XGBoost (completed)

**23 experiments, ~3 hours**

Swapped MLP for XGBoost on same event-window sector-relative setup.

- **Baseline:** 53.6% (XGBoost baseline already higher than MLP's best)
- **Best: 54.5%** (+0.9pp, +0.5pp above MLP's best-ever)

### What worked with XGBoost

1. **Sales growth YoY + operating margin change** — replaced broken surprise features → 53.9%
2. **Transcript tone with post-earnings activation window** — management sentiment, guidance confidence, Q&A intensity → 54.3%
3. **Removing stale long-horizon features** — dropped 20d return, 60d volatility, 60d price position, 200d MA → 54.5%
4. **Transcript Q&A intensity** (analyst_share × log word count) and **analyst sentiment** — lateral improvements at 54.5%

### What didn't work

- Operating cash flow margin, ROA change, accrual ratio
- Sales yield valuation features
- Fiscal-period-aware consensus revisions
- Operating income growth, asset turnover, sales growth acceleration
- Various transcript metadata pruning attempts

### XGBoost feature importance (top 10)

1. estimate_revision_30d
2. analyst_count
3. volatility_10d
4. return_10d
5. estimate_revision_90d
6. macd_signal
7. report_lag_vs_trailing
8. return_3d
9. price_position_60d
10. price_vs_ma_20d

### Key finding: model matters

XGBoost's baseline (53.6%) was already higher than MLP's best after 81 experiments (54.0%). The features discovered by the autoresearch loop contain real signal — the MLP just couldn't extract it efficiently. GBTs are the right model for tabular financial data at this scale.

### Key finding: feature pruning matters in event windows

Removing long-horizon technical features (20d return, 60d vol, 200d MA) improved accuracy. In a 10-day post-earnings window, slow macro/trend state is stale noise. The agent's postmortem: "samples are restricted to the ~10-day post-earnings window, so slow macro/trend state is often stale relative to the current event."

---

## Summary of all runs

| Run | Target | Model | Baseline | Best | Improvement | Experiments |
|-----|--------|-------|----------|------|-------------|-------------|
| 1 | Raw direction | MLP | 52.7% | 53.0% | +0.3pp | 45 |
| 2 | Sector-relative | MLP | 51.5% | 52.2% | +0.7pp | 66 |
| 3 | Event-window sector-relative | MLP | 52.3% | 54.0% | +1.7pp | 81 |
| 4 | Event-window sector-relative | XGBoost | 53.6% | 54.5% | +0.9pp | 23 |

**Total: ~215 experiments, ~$400-600 API cost, ~36 hours runtime**

---

## Setup (current)

- **Model**: XGBoost (500 estimators, depth 4, LR 0.05, early stopping 50)
- **Data**: 30 liquid stocks across 6 sectors, 8yr train / 2yr val, event-window filtered (10 days post-earnings)
- **Target**: Binary sector-relative direction (outperform sector in event window)
- **Evaluation**: 3-run averaged val_accuracy, single walk-forward split
- **Agent**: GPT-5.4 via OpenAI Responses API
- **FactSet access**: Estimates, Fundamentals, Transcripts, Prices SDKs

---

## Next steps to explore

### 1. Conditional prediction with abstention (highest priority)
Only trade when XGBoost's predicted probability is above 60% or below 40%. Could push hit rate to 58-60% on fewer but higher-conviction predictions. Most directly useful for actual trading. Quick to implement — just filter predictions by confidence threshold.

### 2. Rolling walk-forward validation
Current 54.5% is on a single 2-year window. Rolling walk-forward across multiple regimes would validate whether this is real or regime-specific. Essential before trading real money.

### 3. Single sector focus (tech only)
10 tech stocks where FactSet features are most homogeneous. Cross-sector heterogeneity adds noise — FCF margin means different things for JPM vs NVDA.

### 4. Ticker/sector encoding
Add categorical sector feature so XGBoost can learn sector-specific splits (e.g., transcripts matter more for tech than utilities).

### 5. Multi-horizon targets
Predict 1/5/20-day forward excess returns instead of binary direction. More aligned with how quant funds actually trade.

---

## Architecture evolution history

1. **Three layers**: Feature research → Architecture search → Hyperparameter tuning (original design)
2. **Two layers**: Feature research → Hyperparameter tuning (architecture locked — simplicity won)
3. **One layer**: Feature research only (hyperparameters locked — exhausted in ~20 experiments)
4. **XGBoost swap**: Immediately beat MLP by 2.6pp on same features — confirmed features have signal MLP couldn't extract

## Key research findings

1. **Cash-flow fundamentals predict stock-specific alpha** — FCF yield, operating margin changes, and operating leverage deltas are the most robust FactSet-derived features across all runs
2. **Transcript tone predicts sector-relative outperformance** — management sentiment, Q&A intensity, and guidance confidence add signal for sector-relative prediction but not for raw direction
3. **Feature pruning is as valuable as feature addition** — removing stale/noisy features (long-horizon technicals, consensus revisions) consistently improved accuracy
4. **Event-window filtering concentrates signal** — restricting to post-earnings days where fundamental features are fresh improves accuracy by ~2pp
5. **Model matters for tabular data** — XGBoost extracts significantly more signal than MLP from the same features
6. **The autoresearch loop works** — autonomous feature research with postmortems produces genuine, actionable insights about what data predicts financial outcomes
