# Multi-Layer Autoresearch — Research Notes

## Current Approach (April 2026)

### What we're testing

A single-layer autoresearch loop focused on **feature research** for daily stock direction prediction. The agent autonomously proposes features, fetches data from FactSet, tests each feature against a locked model, and keeps/discards based on validation accuracy. Postmortems every 10 experiments let the agent learn what types of features work.

### Setup

- **Model**: Simple MLP [64, 32] with LayerNorm, GELU, Dropout — locked, not modified
- **Hyperparameters**: Locked (AdamW, LR=1e-3, batch=256, dropout=0.2, constant schedule)
- **Data**: 30 liquid stocks across 6 sectors, 8 years training / 2 years validation
- **Target**: Binary daily close direction (up/down)
- **Evaluation**: 3-run averaged val_accuracy to reduce noise
- **Agent**: GPT-5.4 via OpenAI Responses API
- **Experiment budget**: 500 experiments (~2-3 days, ~$500-1000 API cost)

### Key decisions made

1. **Architecture locked**: Layer 2 (architecture search) was removed after overnight testing showed the Deep & Cross + grouped attention model didn't meaningfully outperform a simple MLP. The agent's own postmortem concluded "this dataset rewards constrained inductive bias more than raw architectural complexity."

2. **Hyperparameters locked**: Layer 3 (parameter tuning) was removed after 50+ experiments consistently converged to the same optimum (constant LR=1e-3, batch=1536, dropout=0.2). The search space exhausted in ~20 experiments.

3. **Focus on features**: The real question is what data predicts daily direction, not what model processes it. The agent has full FactSet SDK access to explore fundamental data.

### What's been found so far (20 experiments)

**Features that helped (kept):**
- Sales growth / acceleration + net cash + EV/sales + filing recency
- Cash flow quality: FCF margin, capex intensity, cash conversion
- Operating leverage deltas: gross margin YoY change, sales/GP growth vs overhead growth

**Features that didn't help (discarded):**
- Transcript sentiment (too sparse, inconsistent across sectors)
- Share dilution/buyback (noisy quarter-to-quarter)
- Profitability ratios (ROA, margins — too sector-dependent)
- Daily mark-to-market valuation (reintroduces price noise)
- Accrual quality, working capital, balance sheet stress
- Consensus disagreement / dispersion
- Post-report drift interactions
- Reporting timeliness
- Capital allocation, financing coverage, book value compounding
- Segment concentration, operating drop-through

**Agent's key insight**: "The most robust signal comes from PIT-safe, low-frequency, economically interpretable fundamentals tied to realized operating performance — especially cash generation and operating-leverage changes."

### Known issues to fix in this run

1. `last_surprise_pct` and `avg_surprise_pct` are ALL ZEROS — the earnings surprise computation is broken. The agent is instructed to diagnose and fix this on experiment 0.
2. Agent has never tried removing features — only adding. Now explicitly told it can remove to reduce noise.

### Current accuracy

- Baseline (price/volume features only): ~52.2%
- Best after 20 feature experiments: ~53.0%
- Up-day fraction (naive baseline): ~52.7%

---

## Next steps if accuracy doesn't improve significantly

If after 100+ experiments with fixed surprise features and feature removal, accuracy remains stuck at ~53%:

### 1. Add ticker/sector encoding
The model currently doesn't know which stock it's looking at. Adding a ticker embedding or sector one-hot encoding would let it learn stock-specific or sector-specific patterns. Easy to implement — just add a categorical feature.

### 2. Focus on a single sector
Instead of 30 stocks across 6 sectors, pick one sector (e.g., large-cap tech: AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA, CRM, ADBE, ORCL). Stocks within a sector have more correlated behavior, so features that work for one should work for others. Reduces noise from sector heterogeneity.

### 3. Change the target variable
Daily binary direction on liquid large-caps may be genuinely too efficient to predict. Alternatives:
- **Weekly direction** — less noise, still abundant data
- **Sector-relative direction** — predict if stock outperforms its sector (removes market beta)
- **Conditional prediction** — only predict on high-conviction days, abstain otherwise
- **Multi-horizon forward excess returns** — what the ML engineer recommended

### 4. Try gradient boosted trees
The ML literature consistently shows XGBoost/LightGBM beats neural networks on tabular data of this size. If the features we discovered are genuinely predictive, a GBT might extract more signal from them than a simple MLP. This would be a separate experiment — keep the same features, swap the model.

### 5. Event-driven subsystem
Instead of daily prediction on all days, focus on **earnings announcement days** where the FactSet data (estimate revisions, surprise history, transcript sentiment) is most relevant. The hedge fund system already showed this works — 75% training hit rate on earnings-driven signals.

### 6. Longer run with accumulated knowledge
Don't reset the runtime between runs. Let postmortem knowledge accumulate across multiple sessions so the agent builds a deeper understanding of what works over 200+ experiments.

---

## Architecture evolution history

1. **Three layers**: Feature research → Architecture search → Hyperparameter tuning (original design)
2. **Two layers**: Feature research → Hyperparameter tuning (architecture locked after overnight run showed simplicity wins)
3. **One layer**: Feature research only (hyperparameters locked after 50+ experiments converged to same optimum)

This progression validated that for this specific task (daily direction prediction with tabular features), the model and hyperparameters matter much less than what data you feed in.
