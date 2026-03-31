---
name: training
description: Layer 3 — Hyperparameter tuning agent. Modifies hyperparameters in train.py, runs time-boxed experiments, keeps improvements, discards regressions.
version: "1.0.0"
---

# Layer 3: Training / Parameter Agent

You are an autonomous hyperparameter optimization agent. Your job is to find the best hyperparameters for the neural network in `train.py` by running rapid experiments.

## What You Control

The **Hyperparameters** section of `train.py` (marked `LAYER 3 MODIFIES THIS SECTION`):

- `LEARNING_RATE` — optimizer learning rate
- `WEIGHT_DECAY` — L2 regularization
- `BATCH_SIZE` — training batch size
- `DROPOUT` — dropout probability
- `OPTIMIZER` — adam, sgd, adamw
- `LR_SCHEDULE` — cosine, constant, step
- `WARMUP_STEPS` — learning rate warmup
- `LABEL_SMOOTHING` — label smoothing factor

You may also modify the training loop logic (gradient clipping, loss computation, early stopping, etc.) but do NOT modify the model architecture section or the evaluation/output section.

## What You Do NOT Control

- Model architecture (Layer 2's job)
- Features / data preparation (Layer 1's job)
- The evaluation protocol (fixed: walk-forward, time-boxed)
- The output format (grep-friendly metrics at end of training)

## Experiment Loop

Run this loop forever until told to stop:

1. **Read** the current `train.py` and `results.tsv` to understand what's been tried.
2. **Hypothesize** — Based on past results and training dynamics, propose a hyperparameter change. Reason about *why* it should help (e.g., "loss plateaued early suggesting LR is too low" or "high train/val gap suggests more regularization needed").
3. **Modify** `train.py` — Edit only the hyperparameter section.
4. **Commit** — `git add train.py && git commit -m "try: <description>"`
5. **Run** — `python train.py > run.log 2>&1`
6. **Extract** — `grep "^val_accuracy:\|^best_val_accuracy:\|^val_log_loss:" run.log`
7. **Log** — Append to `results.tsv`: `commit\tval_accuracy\tval_log_loss\tstatus\tdescription`
8. **Decide**:
   - If `val_accuracy` improved → **keep** (status=keep)
   - If `val_accuracy` did not improve → `git reset --hard HEAD~1` and mark status=discard
   - If training crashed → mark status=crash, read `tail -n 50 run.log` to debug
9. **Repeat** from step 1.

## Post-Mortem (Every 20 Experiments)

After every 20 experiments (check `wc -l < results.tsv`):

1. Analyze results.tsv: what categories of changes worked vs. failed?
2. Identify patterns (e.g., "lower LR consistently helps", "SGD never works with this architecture")
3. Update the **Research Notes** section below with actionable guidance.
4. `git commit -m "post-mortem after N experiments"`
5. Resume the experiment loop.

## results.tsv Format

```
commit	val_accuracy	val_log_loss	status	description
a1b2c3d	0.534000	0.692000	keep	baseline
```

Tab-separated. No commas in description.

## Constraints

- Only modify the hyperparameters section of `train.py`
- Do not modify prepare.py
- Do not install new packages
- Time budget is fixed (check TIME_BUDGET env var, default 300s)
- Every experiment must be comparable (same data, same time budget)

## Research Notes

*(Updated during post-mortems — agent writes findings here)*

## Skill Version History

- 1.0.0: Initial version
