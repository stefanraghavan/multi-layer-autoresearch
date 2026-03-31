"""Phase runners for each layer's experiment and postmortem cycles."""

from __future__ import annotations

from pathlib import Path

from lib.agent import run_agent
from lib.config import (
    AgentExecution,
    EXPERIMENT_TOOLS,
    POSTMORTEM_TOOLS,
    LAYER1_POSTMORTEM_EVERY,
    LAYER2_POSTMORTEM_EVERY,
    LAYER3_POSTMORTEM_EVERY,
)
from lib.utils import _log


# ---------------------------------------------------------------------------
# Layer 3: Training / Parameters
# ---------------------------------------------------------------------------


async def run_layer3_experiment(
    runtime_dir: Path,
    experiment_num: int,
    best_val_accuracy: float,
) -> tuple[str, AgentExecution]:
    """Run a single Layer 3 hyperparameter experiment."""

    prompt = f"""You are the Layer 3 (Training/Parameters) agent. Follow the training skill.

Current state:
- Experiment number: {experiment_num}
- Best val_accuracy so far: {best_val_accuracy:.6f}
- Postmortem due: {"YES — do post-mortem first" if experiment_num > 0 and experiment_num % LAYER3_POSTMORTEM_EVERY == 0 else "no"}

Instructions:
1. Read `train.py` and `results.tsv` (create results.tsv with header if it doesn't exist).
2. Propose ONE hyperparameter change to `train.py` (only the hyperparameters section).
3. Commit, run `python train.py > run.log 2>&1`, extract metrics, log to results.tsv.
4. If val_accuracy > {best_val_accuracy:.6f}, keep. Otherwise revert with `git reset --hard HEAD~1`.
5. Report the result.

{"IMPORTANT: Post-mortem is due. Before running the next experiment, analyze results.tsv, identify patterns, and update the Research Notes section in .claude/skills/training/SKILL.md. Commit the post-mortem update." if experiment_num > 0 and experiment_num % LAYER3_POSTMORTEM_EVERY == 0 else ""}

Output your result as:
RESULT: val_accuracy=<value> status=<keep|discard|crash> description=<what you tried>
"""

    stats = await run_agent(runtime_dir, prompt, EXPERIMENT_TOOLS)
    return stats.result_text, stats


# ---------------------------------------------------------------------------
# Layer 2: Architecture
# ---------------------------------------------------------------------------


async def run_layer2_experiment(
    runtime_dir: Path,
    experiment_num: int,
    best_val_accuracy: float,
) -> tuple[str, AgentExecution]:
    """Run a single Layer 2 architecture experiment."""

    prompt = f"""You are the Layer 2 (Architecture) agent. Follow the architecture skill.

Current state:
- Architecture experiment number: {experiment_num}
- Best val_accuracy so far: {best_val_accuracy:.6f}
- Postmortem due: {"YES — do post-mortem first" if experiment_num > 0 and experiment_num % LAYER2_POSTMORTEM_EVERY == 0 else "no"}

Instructions:
1. Read `train.py` and `results.tsv`.
2. Propose ONE architecture change to `train.py` (model architecture section and/or StockPredictor class).
3. Reset hyperparameters to reasonable defaults if the architecture change is major.
4. Commit, run `python train.py > run.log 2>&1`, extract metrics, log to results.tsv.
5. If val_accuracy > {best_val_accuracy:.6f}, keep. Otherwise revert with `git reset --hard HEAD~1`.
6. Report the result.

{"IMPORTANT: Post-mortem is due. Before running the next experiment, analyze results.tsv for architecture patterns, and update the Research Notes section in .claude/skills/architecture/SKILL.md. Commit the post-mortem update." if experiment_num > 0 and experiment_num % LAYER2_POSTMORTEM_EVERY == 0 else ""}

Output your result as:
RESULT: val_accuracy=<value> status=<keep|discard|crash> description=<what you tried>
"""

    stats = await run_agent(runtime_dir, prompt, EXPERIMENT_TOOLS)
    return stats.result_text, stats


# ---------------------------------------------------------------------------
# Layer 1: Feature Research
# ---------------------------------------------------------------------------


async def run_layer1_experiment(
    runtime_dir: Path,
    experiment_num: int,
    best_val_accuracy: float,
) -> tuple[str, AgentExecution]:
    """Run a single Layer 1 feature research experiment."""

    prompt = f"""You are the Layer 1 (Feature Research) agent. Follow the feature-research skill.

Current state:
- Feature experiment number: {experiment_num}
- Best val_accuracy so far: {best_val_accuracy:.6f}
- Postmortem due: {"YES — do post-mortem first" if experiment_num > 0 and experiment_num % LAYER1_POSTMORTEM_EVERY == 0 else "no"}

Instructions:
1. Read `prepare.py` and `results.tsv`.
2. Propose ONE feature change to `prepare.py` (the compute_all_features function or add new feature functions).
3. Commit prepare.py.
4. Re-run data prep: `python prepare.py`
5. Run training: `python train.py > run.log 2>&1`
6. Extract metrics, log to results.tsv.
7. If val_accuracy > {best_val_accuracy:.6f}, keep. Otherwise revert with `git reset --hard HEAD~1`.
8. Report the result.

Remember: all features must be point-in-time. No future data leakage.

{"IMPORTANT: Post-mortem is due. Before running the next experiment, analyze results.tsv for feature patterns, and update the Research Notes section in .claude/skills/feature-research/SKILL.md. Commit the post-mortem update." if experiment_num > 0 and experiment_num % LAYER1_POSTMORTEM_EVERY == 0 else ""}

Output your result as:
RESULT: val_accuracy=<value> status=<keep|discard|crash> description=<what you tried>
"""

    stats = await run_agent(runtime_dir, prompt, EXPERIMENT_TOOLS)
    return stats.result_text, stats


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def parse_result_text(result_text: str) -> dict:
    """Extract structured result from agent's output text."""
    result = {
        "val_accuracy": None,
        "status": "crash",
        "description": "unknown",
    }

    for line in result_text.splitlines():
        line = line.strip()
        if line.startswith("RESULT:"):
            parts = line[len("RESULT:"):].strip()
            for part in parts.split():
                if "=" in part:
                    key, value = part.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key == "val_accuracy":
                        try:
                            result["val_accuracy"] = float(value)
                        except ValueError:
                            pass
                    elif key == "status":
                        result["status"] = value
                    elif key == "description":
                        result["description"] = value
            break

    return result
