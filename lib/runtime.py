"""Runtime environment initialization: directory setup, skill copying, CLAUDE.md generation."""

from __future__ import annotations

import shutil
from pathlib import Path

from lib.paths import (
    runtime_dir,
    runtime_skills_dir,
    source_skills_dir,
    trace_dir,
)


def _default_claude_template() -> str:
    return """# CLAUDE.md - Multi-Layer Autoresearch Runtime Context

You are working in an autonomous research environment for daily stock direction prediction.

## System Overview

This system uses three nested layers of autonomous research:
- **Layer 1 (Feature Research)**: Determines what features/data to include
- **Layer 2 (Architecture)**: Determines the neural network architecture
- **Layer 3 (Training/Parameters)**: Tunes hyperparameters and trains the model

## Skills
- `.claude/skills/feature-research/SKILL.md` — Feature research agent
- `.claude/skills/architecture/SKILL.md` — Architecture agent
- `.claude/skills/training/SKILL.md` — Training/parameter agent

## Directory Conventions
- `workspace/{run_id}/` for run-scoped scratch work
- `traces/{run_id}.json` for experiment outputs
- `data/` for prepared datasets
- `train.py` is the training script (modified by Layer 2 and Layer 3)
- `prepare.py` is the data preparation script (modified by Layer 1 for features)
- `results.tsv` for experiment logging

## Key Files
- `train.py` — Neural network training script (Layer 2 modifies architecture, Layer 3 modifies hyperparameters)
- `prepare.py` — Data preparation and feature engineering (Layer 1 modifies feature set)
- `features.json` — Current feature configuration (Layer 1 output)
- `architecture.json` — Current architecture configuration (Layer 2 output)
"""


def generate_runtime_claude_md(rt_dir: Path) -> None:
    (rt_dir / "CLAUDE.md").write_text(_default_claude_template())


def initialize_runtime(reset: bool = False) -> Path:
    """Initialize the runtime environment.

    On first run: copies baseline skills, train.py, prepare.py into runtime.
    On subsequent runs: preserves evolved skills and code (postmortem improvements
    accumulate across runs). Only resets if reset=True or RESET_RUNTIME=true.

    Runtime lives at ~/multi-layer-autoresearch-data/runtime/ (outside the repo).
    """
    import os
    reset = reset or os.environ.get("RESET_RUNTIME", "false").strip().lower() in {"1", "true", "yes"}

    rt_dir = runtime_dir()
    is_fresh = not rt_dir.exists()
    rt_dir.mkdir(parents=True, exist_ok=True)

    for subdir in ("workspace", "traces", "data"):
        (rt_dir / subdir).mkdir(exist_ok=True)

    src_skills = source_skills_dir()
    dst_skills = runtime_skills_dir()
    dst_skills.parent.mkdir(parents=True, exist_ok=True)

    if is_fresh or reset:
        # First run or explicit reset: copy baseline skills
        for skill_dir in sorted(src_skills.iterdir()):
            if not skill_dir.is_dir():
                continue
            target = dst_skills / skill_dir.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(skill_dir, target)

        # Copy train.py and prepare.py into runtime
        from lib.paths import repo_dir
        for filename in ("train.py", "prepare.py"):
            src = repo_dir() / filename
            dst = rt_dir / filename
            if src.exists():
                shutil.copy2(src, dst)
    else:
        # Subsequent run: only copy skills/code that don't exist yet
        # (preserves evolved skills from postmortems)
        for skill_dir in sorted(src_skills.iterdir()):
            if not skill_dir.is_dir():
                continue
            target = dst_skills / skill_dir.name
            if not target.exists():
                shutil.copytree(skill_dir, target)

        from lib.paths import repo_dir
        for filename in ("train.py", "prepare.py"):
            src = repo_dir() / filename
            dst = rt_dir / filename
            if not dst.exists() and src.exists():
                shutil.copy2(src, dst)

    generate_runtime_claude_md(rt_dir)
    return rt_dir
