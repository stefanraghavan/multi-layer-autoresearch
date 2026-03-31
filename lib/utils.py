"""Small helpers with no domain coupling."""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Any, Optional

from lib.config import VERBOSE


def _log(message: str) -> None:
    if VERBOSE:
        print(message, flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_id(layer: str, experiment_num: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{layer}_exp{experiment_num}_{ts}_{uuid.uuid4().hex[:6]}"


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sum_optional_int(values: list[Optional[int]]) -> Optional[int]:
    concrete = [v for v in values if v is not None]
    return int(sum(concrete)) if concrete else None


def _sum_optional_float(values: list[Optional[float]]) -> Optional[float]:
    concrete = [v for v in values if v is not None]
    return float(sum(concrete)) if concrete else None


def _get_sdk_version() -> Optional[str]:
    try:
        return package_version("openai")
    except PackageNotFoundError:
        return None
