"""Constants, environment loading, and dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _load_local_env() -> None:
    """Load .env from repo root into process env if variables are unset."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


_load_local_env()

# ---------------------------------------------------------------------------
# Core settings
# ---------------------------------------------------------------------------

MODEL_ID = os.environ.get("MODEL_ID", "gpt-5.4")
VERBOSE = os.environ.get("VERBOSE", "true").strip().lower() not in {
    "0", "false", "no",
}

# ---------------------------------------------------------------------------
# Stock universe
# ---------------------------------------------------------------------------

# Comma-separated list of tickers to include in the dataset.
# Default: 30 liquid stocks with FactSet coverage across sectors
DEFAULT_TICKERS = "AAPL,MSFT,AMZN,GOOGL,META,NVDA,TSLA,CRM,ADBE,ORCL,JPM,BAC,GS,MS,V,MA,JNJ,UNH,PFE,ABT,WMT,HD,MCD,NKE,CAT,BA,XOM,CVX,NEE,UPS"
TICKERS = [
    t.strip().upper()
    for t in os.environ.get("TICKERS", DEFAULT_TICKERS).split(",")
    if t.strip()
]

# Walk-forward settings
TRAIN_YEARS = int(os.environ.get("TRAIN_YEARS", "8"))
VALIDATION_YEARS = int(os.environ.get("VALIDATION_YEARS", "2"))

# ---------------------------------------------------------------------------
# Experiment budgets
# ---------------------------------------------------------------------------

# Number of feature experiments per run
LAYER1_EXPERIMENTS_PER_CYCLE = int(os.environ.get("NUM_EXPERIMENTS", "500"))
# Postmortem frequency (every N experiments)
LAYER1_POSTMORTEM_EVERY = int(os.environ.get("POSTMORTEM_EVERY", "10"))

# ---------------------------------------------------------------------------
# Tools configuration
# ---------------------------------------------------------------------------

EXPERIMENT_TOOLS = ["Skill", "Read", "Write", "Edit", "Glob", "Grep", "Bash"]
POSTMORTEM_TOOLS = ["Skill", "Read", "Write", "Edit", "Glob", "Grep"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AgentExecution:
    result_text: str
    duration_ms: int
    tokens_in: Optional[int]
    tokens_out: Optional[int]
    cost_usd: Optional[float]
