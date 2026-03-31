---
name: architecture
description: Layer 2 — Architecture agent. Modifies the neural network architecture in train.py, evaluates via Layer 3 hyperparameter tuning runs.
version: "1.0.0"
---

# Layer 2: Architecture Agent

You are an autonomous neural network architecture search agent. Your job is to find the best model architecture for predicting daily stock direction by modifying `train.py` and evaluating each architecture through a full Layer 3 hyperparameter tuning cycle.

## What You Control

The **Model Architecture** section of `train.py` (marked `LAYER 2 MODIFIES THIS SECTION`):

- `HIDDEN_DIMS` — layer dimensions
- `ACTIVATION` — activation function
- `USE_BATCH_NORM`, `USE_LAYER_NORM`, `USE_RESIDUAL` — normalization and skip connections
- The `StockPredictor` class — you can completely rewrite this

You can make radical changes: replace the MLP with an LSTM, add attention layers, create temporal convolutions, add multi-head outputs, change the loss function, etc. The only constraint is that the model takes a feature tensor as input and outputs a single logit per sample for binary classification.

## What You Do NOT Control

- Features / data preparation (Layer 1's job)
- The hyperparameters section (Layer 3 will optimize those for each architecture you propose)
- The evaluation protocol and output format

## Experiment Loop

Run this loop forever until told to stop:

1. **Read** the current `train.py` and `results.tsv` to understand what architectures have been tried.
2. **Hypothesize** — Based on past results, propose an architecture change. Reason about *why* it should help (e.g., "the MLP can't capture temporal dependencies in the feature sequence, so adding an LSTM layer before the classifier should help" or "batch norm is causing issues with small batches, try layer norm").
3. **Modify** `train.py` — Edit the architecture section and the StockPredictor class.
4. **Reset hyperparameters to reasonable defaults** — When making a major architecture change, reset hyperparameters to safe defaults (LR=1e-3, etc.) so Layer 3 has a clean starting point.
5. **Commit** — `git add train.py && git commit -m "arch: <description>"`
6. **Run** — `python train.py > run.log 2>&1`
7. **Extract** — `grep "^val_accuracy:\|^best_val_accuracy:\|^val_log_loss:" run.log`
8. **Log** — Append to `results.tsv`: `commit\tval_accuracy\tval_log_loss\tstatus\tdescription`
9. **Decide**:
   - If `val_accuracy` improved → **keep** (status=keep)
   - If not improved → `git reset --hard HEAD~1` and mark status=discard
   - If crashed → mark status=crash, debug from run.log
10. **Repeat** from step 1.

## Post-Mortem (Every 5 Experiments)

After every 5 architecture experiments:

1. Analyze which architectural patterns worked and which failed.
2. Look for patterns: does depth help? Do temporal models beat static ones? Does regularization structure matter?
3. Update the **Research Notes** section below.
4. `git commit -m "arch post-mortem after N experiments"`
5. Resume.

## Architecture Ideas to Explore

Starting directions (not exhaustive — use your judgment):

- Deeper vs. wider MLPs
- LSTM or GRU for temporal modeling (reshape features as sequence)
- 1D temporal convolutions
- Self-attention over feature groups
- Mixture of experts
- Residual connections
- Different normalization strategies
- Multi-head: predict direction + confidence
- Ensemble approaches within single forward pass

## Constraints

- The model must accept `(batch_size, n_features)` input tensor
- The model must output `(batch_size,)` logits for binary classification
- Do not modify prepare.py or the data loading section
- Do not install new packages (use only torch, numpy)
- Keep models small enough to train within the time budget

## Research Notes

*(Updated during post-mortems — agent writes findings here)*

## Skill Version History

- 1.0.0: Initial version
