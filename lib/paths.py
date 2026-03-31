"""Path helpers for runtime and repository structure."""

from __future__ import annotations

import os
from pathlib import Path


def repo_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    default = str(Path.home() / "multi-layer-autoresearch-data")
    return Path(os.environ.get("DATA_DIR", default)).expanduser()


def runtime_dir() -> Path:
    return data_dir() / "runtime"


def runtime_skills_dir() -> Path:
    return runtime_dir() / ".claude" / "skills"


def source_skills_dir() -> Path:
    primary = repo_dir() / "skills"
    if primary.exists():
        return primary
    raise FileNotFoundError("No baseline skills directory found at skills/.")


def trace_dir() -> Path:
    return runtime_dir() / "traces"


def workspace_dir(run_id: str) -> Path:
    return runtime_dir() / "workspace" / run_id


def db_path() -> Path:
    return data_dir() / "db" / "training.db"
