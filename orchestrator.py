"""Orchestrator: runs the feature research autoresearch loop.

Single layer: the agent iterates on features in prepare.py.
Architecture and hyperparameters are locked.

Usage:
    python orchestrator.py                          # Run feature research loop
    NUM_EXPERIMENTS=50 python orchestrator.py        # Override experiment count
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from lib.config import (
    MODEL_ID,
    TICKERS,
    LAYER1_EXPERIMENTS_PER_CYCLE,
)
from lib.db import (
    init_database,
    insert_experiment,
    insert_results_log,
)
from lib.phases import (
    run_feature_experiment,
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
        timeout=600,
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


def _copy_factset_cache(runtime_dir: Path) -> None:
    """Copy FactSet cached data from repo data/ into runtime data/ so prepare.py can find it."""
    import shutil
    from lib.paths import repo_dir

    for source_base in [repo_dir() / "data", Path(__file__).parent / "data"]:
        if not source_base.exists():
            continue
        for ticker_dir in source_base.iterdir():
            if not ticker_dir.is_dir():
                continue
            for cache_name in ["factset_cache.json", "factset_fundamentals_qtr.json",
                               "factset_profitability_qtr.json", "factset_transcripts.json"]:
                cache_file = ticker_dir / cache_name
                if cache_file.exists():
                    dest_dir = runtime_dir / "data" / ticker_dir.name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_file = dest_dir / cache_name
                    if not dest_file.exists():
                        shutil.copy2(cache_file, dest_file)
    _log("  FactSet caches copied.")


# ---------------------------------------------------------------------------
# Feature research loop
# ---------------------------------------------------------------------------


async def run_feature_loop(runtime_dir: Path, n_experiments: int, baseline_accuracy: float) -> float:
    """Run the feature research loop for N experiments. Returns best val_accuracy."""
    _log(f"\n{'='*60}")
    _log(f"FEATURE RESEARCH: Running {n_experiments} experiments")
    _log(f"  Baseline to beat: {baseline_accuracy:.4f}")
    _log(f"{'='*60}")

    best_accuracy = baseline_accuracy

    for i in range(n_experiments):
        run = _run_id("feat", i)
        _log(f"\n--- Experiment {i+1}/{n_experiments} (best={best_accuracy:.4f}) ---")

        try:
            result_text, stats = await run_feature_experiment(
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

            _log(f"  Result: val_accuracy={val_acc}, status={status}, description={description}")

        except Exception as exc:
            _log(f"  !! Experiment {i} failed: {exc}")
            insert_experiment(
                run_id=run,
                layer="feature",
                description=f"error: {exc}",
                status="crash",
                error_message=str(exc),
            )

    _log(f"\nFeature research complete. Best accuracy: {best_accuracy:.4f}")
    return best_accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    n_experiments = int(os.environ.get("NUM_EXPERIMENTS", str(LAYER1_EXPERIMENTS_PER_CYCLE)))

    _log("=" * 60)
    _log("Feature Research Autoresearch")
    _log(f"  Model: {MODEL_ID}")
    _log(f"  Tickers: {TICKERS}")
    _log(f"  Experiments: {n_experiments}")
    _log("=" * 60)

    # Initialize
    init_database()
    rt_dir = initialize_runtime()
    _log(f"Runtime: {rt_dir}")

    # Copy all FactSet caches into runtime
    _copy_factset_cache(rt_dir)

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

    # Run feature research loop
    started = perf_counter()
    best = await run_feature_loop(rt_dir, n_experiments, baseline_accuracy)
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
