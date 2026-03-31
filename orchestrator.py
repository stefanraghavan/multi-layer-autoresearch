"""Orchestrator: runs the three-layer nested autoresearch loop.

Layer 1 (Feature Research) is the outermost loop.
  For each feature experiment, Layer 2 (Architecture) runs N architecture experiments.
    For each architecture experiment, Layer 3 (Training) runs M hyperparameter experiments.

Usage:
    python orchestrator.py                          # Full three-layer loop
    LAYER=3 python orchestrator.py                  # Layer 3 only (fixed features + arch)
    LAYER=2 python orchestrator.py                  # Layer 2 + 3 (fixed features)
    TICKERS=SPY python orchestrator.py              # Specify ticker(s)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from lib.config import (
    MODEL_ID,
    TICKERS,
    LAYER1_EXPERIMENTS_PER_CYCLE,
    LAYER2_EXPERIMENTS_PER_EVAL,
    LAYER3_EXPERIMENTS_PER_EVAL,
)
from lib.db import (
    init_database,
    insert_experiment,
    insert_results_log,
)
from lib.phases import (
    run_layer1_experiment,
    run_layer2_experiment,
    run_layer3_experiment,
    parse_result_text,
)
from lib.runtime import initialize_runtime
from lib.utils import _log, _run_id


def _prepare_data(runtime_dir: Path) -> None:
    """Run prepare.py to generate the dataset."""
    _log("Preparing data...")
    result = subprocess.run(
        [sys.executable, "prepare.py"],
        cwd=str(runtime_dir),
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "DATA_DIR": str(runtime_dir / "data")},
    )
    if result.returncode != 0:
        _log(f"  !! prepare.py failed:\n{result.stderr[-2000:]}")
        raise RuntimeError("Data preparation failed")
    _log(f"  Data prepared successfully.")


def _init_git(runtime_dir: Path) -> None:
    """Initialize git repo in runtime directory for version control."""
    git_dir = runtime_dir / ".git"
    if not git_dir.exists():
        subprocess.run(["git", "init"], cwd=str(runtime_dir), capture_output=True)
        subprocess.run(["git", "add", "-A"], cwd=str(runtime_dir), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial baseline"],
            cwd=str(runtime_dir),
            capture_output=True,
        )
        _log("  Git repo initialized.")


def _extract_accuracy_from_result(result_text: str) -> float | None:
    """Parse val_accuracy from agent result text."""
    parsed = parse_result_text(result_text)
    return parsed.get("val_accuracy")


# ---------------------------------------------------------------------------
# Layer runners
# ---------------------------------------------------------------------------


async def run_layer3_loop(runtime_dir: Path, n_experiments: int, baseline_accuracy: float = 0.0) -> float:
    """Run Layer 3 (hyperparameter tuning) for N experiments. Returns best val_accuracy."""
    _log(f"\n{'='*60}")
    _log(f"LAYER 3: Running {n_experiments} hyperparameter experiments")
    _log(f"  Baseline to beat: {baseline_accuracy:.4f}")
    _log(f"{'='*60}")

    best_accuracy = baseline_accuracy

    for i in range(n_experiments):
        run = _run_id("L3", i)
        _log(f"\n--- Layer 3, Experiment {i+1}/{n_experiments} (best={best_accuracy:.4f}) ---")

        try:
            result_text, stats = await run_layer3_experiment(
                runtime_dir, i, best_accuracy,
            )
            parsed = parse_result_text(result_text)
            val_acc = parsed.get("val_accuracy")
            status = parsed.get("status", "crash")
            description = parsed.get("description", "unknown")

            if val_acc is not None and val_acc > best_accuracy:
                best_accuracy = val_acc

            insert_experiment(
                run_id=run,
                layer="training",
                description=description,
                accuracy=val_acc,
                status=status,
                duration_ms=stats.duration_ms,
                tokens_in=stats.tokens_in,
                tokens_out=stats.tokens_out,
                cost_usd=stats.cost_usd,
                model_id=MODEL_ID,
            )
            insert_results_log("training", run, val_acc, "accuracy", status, description)

            _log(f"  Result: val_accuracy={val_acc}, status={status}")

        except Exception as exc:
            _log(f"  !! Layer 3 experiment {i} failed: {exc}")
            insert_experiment(
                run_id=run,
                layer="training",
                description=f"error: {exc}",
                status="crash",
                error_message=str(exc),
            )

    _log(f"\nLayer 3 complete. Best accuracy: {best_accuracy:.4f}")
    return best_accuracy


async def run_layer2_loop(runtime_dir: Path, n_experiments: int, baseline_accuracy: float = 0.0) -> float:
    """Run Layer 2 (architecture search) for N experiments.
    Each experiment includes a full Layer 3 tuning cycle.
    Returns best val_accuracy.
    """
    _log(f"\n{'='*60}")
    _log(f"LAYER 2: Running {n_experiments} architecture experiments")
    _log(f"  (each with {LAYER3_EXPERIMENTS_PER_EVAL} Layer 3 runs)")
    _log(f"  Baseline to beat: {baseline_accuracy:.4f}")
    _log(f"{'='*60}")

    best_accuracy = baseline_accuracy

    for i in range(n_experiments):
        run = _run_id("L2", i)
        _log(f"\n=== Layer 2, Experiment {i+1}/{n_experiments} (best={best_accuracy:.4f}) ===")

        try:
            # Layer 2 agent proposes an architecture change
            result_text, stats = await run_layer2_experiment(
                runtime_dir, i, best_accuracy,
            )
            parsed = parse_result_text(result_text)
            val_acc = parsed.get("val_accuracy")
            status = parsed.get("status", "crash")
            description = parsed.get("description", "unknown")

            # After architecture change, run Layer 3 to tune hyperparameters
            if status == "keep":
                _log(f"  Architecture kept. Running Layer 3 tuning...")
                l3_best = await run_layer3_loop(runtime_dir, LAYER3_EXPERIMENTS_PER_EVAL)
                if l3_best > (val_acc or 0):
                    val_acc = l3_best

            if val_acc is not None and val_acc > best_accuracy:
                best_accuracy = val_acc

            insert_experiment(
                run_id=run,
                layer="architecture",
                description=description,
                accuracy=val_acc,
                status=status,
                duration_ms=stats.duration_ms,
                tokens_in=stats.tokens_in,
                tokens_out=stats.tokens_out,
                cost_usd=stats.cost_usd,
                model_id=MODEL_ID,
            )
            insert_results_log("architecture", run, val_acc, "accuracy", status, description)

            _log(f"  Architecture result: val_accuracy={val_acc}, status={status}")

        except Exception as exc:
            _log(f"  !! Layer 2 experiment {i} failed: {exc}")
            insert_experiment(
                run_id=run,
                layer="architecture",
                description=f"error: {exc}",
                status="crash",
                error_message=str(exc),
            )

    _log(f"\nLayer 2 complete. Best accuracy: {best_accuracy:.4f}")
    return best_accuracy


async def run_layer1_loop(runtime_dir: Path, n_experiments: int, baseline_accuracy: float = 0.0) -> float:
    """Run Layer 1 (feature research) for N experiments.
    Each experiment includes a full Layer 2 + Layer 3 cycle.
    Returns best val_accuracy.
    """
    _log(f"\n{'='*60}")
    _log(f"LAYER 1: Running {n_experiments} feature experiments")
    _log(f"  (each with {LAYER2_EXPERIMENTS_PER_EVAL} Layer 2 + {LAYER3_EXPERIMENTS_PER_EVAL} Layer 3 runs)")
    _log(f"  Baseline to beat: {baseline_accuracy:.4f}")
    _log(f"{'='*60}")

    best_accuracy = baseline_accuracy

    for i in range(n_experiments):
        run = _run_id("L1", i)
        _log(f"\n{'#'*60}")
        _log(f"Layer 1, Experiment {i+1}/{n_experiments} (best={best_accuracy:.4f})")
        _log(f"{'#'*60}")

        try:
            # Layer 1 agent proposes a feature change and re-runs data prep + training
            result_text, stats = await run_layer1_experiment(
                runtime_dir, i, best_accuracy,
            )
            parsed = parse_result_text(result_text)
            val_acc = parsed.get("val_accuracy")
            status = parsed.get("status", "crash")
            description = parsed.get("description", "unknown")

            # After feature change, run Layer 2 + 3 to find best arch + params
            if status == "keep":
                _log(f"  Features kept. Running Layer 2 + 3 optimization...")
                l2_best = await run_layer2_loop(runtime_dir, LAYER2_EXPERIMENTS_PER_EVAL)
                if l2_best > (val_acc or 0):
                    val_acc = l2_best

            if val_acc is not None and val_acc > best_accuracy:
                best_accuracy = val_acc

            insert_experiment(
                run_id=run,
                layer="feature",
                description=description,
                accuracy=val_acc,
                status=status,
                duration_ms=stats.duration_ms,
                tokens_in=stats.tokens_in,
                tokens_out=stats.tokens_out,
                cost_usd=stats.cost_usd,
                model_id=MODEL_ID,
            )
            insert_results_log("feature", run, val_acc, "accuracy", status, description)

            _log(f"  Feature result: val_accuracy={val_acc}, status={status}")

        except Exception as exc:
            _log(f"  !! Layer 1 experiment {i} failed: {exc}")
            insert_experiment(
                run_id=run,
                layer="feature",
                description=f"error: {exc}",
                status="crash",
                error_message=str(exc),
            )

    _log(f"\nLayer 1 complete. Best accuracy: {best_accuracy:.4f}")
    return best_accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    start_layer = int(os.environ.get("LAYER", "1"))

    _log("=" * 60)
    _log("Multi-Layer Autoresearch")
    _log(f"  Model: {MODEL_ID}")
    _log(f"  Tickers: {TICKERS}")
    _log(f"  Starting from Layer: {start_layer}")
    _log("=" * 60)

    # Initialize
    init_database()
    rt_dir = initialize_runtime()
    _log(f"Runtime: {rt_dir}")

    # Prepare data
    _prepare_data(rt_dir)

    # Initialize git for keep/discard mechanism
    _init_git(rt_dir)

    # Run baseline
    _log("\nRunning baseline training...")
    baseline_result = subprocess.run(
        [sys.executable, "train.py"],
        cwd=str(rt_dir),
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "DATA_DIR": str(rt_dir / "data")},
    )
    baseline_accuracy = 0.0
    if baseline_result.returncode == 0:
        for line in baseline_result.stdout.splitlines():
            if line.startswith("val_accuracy:"):
                baseline_accuracy = float(line.split()[-1])
                break
    _log(f"Baseline val_accuracy: {baseline_accuracy:.4f}")

    # Run the appropriate layer loop, seeded with baseline accuracy
    started = perf_counter()

    if start_layer == 3:
        best = await run_layer3_loop(rt_dir, LAYER3_EXPERIMENTS_PER_EVAL, baseline_accuracy)
    elif start_layer == 2:
        best = await run_layer2_loop(rt_dir, LAYER2_EXPERIMENTS_PER_EVAL, baseline_accuracy)
    else:
        best = await run_layer1_loop(rt_dir, LAYER1_EXPERIMENTS_PER_CYCLE, baseline_accuracy)

    elapsed = perf_counter() - started

    _log(f"\n{'='*60}")
    _log(f"COMPLETE")
    _log(f"  Baseline: {baseline_accuracy:.4f}")
    _log(f"  Best:     {best:.4f}")
    _log(f"  Improvement: {best - baseline_accuracy:+.4f}")
    _log(f"  Time: {elapsed/3600:.1f} hours")
    _log(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
