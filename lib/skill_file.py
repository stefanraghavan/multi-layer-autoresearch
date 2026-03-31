"""Utilities for parsing and validating skill markdown files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


def _extract_frontmatter_block(text: str) -> str:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("Missing YAML frontmatter start delimiter '---'.")

    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            return "\n".join(lines[1:idx])

    raise ValueError("Missing YAML frontmatter end delimiter '---'.")


def read_frontmatter(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text()
    block = _extract_frontmatter_block(text)

    if yaml is not None:
        parsed = yaml.safe_load(block) or {}
        if not isinstance(parsed, dict):
            raise ValueError("Frontmatter is not a key/value mapping.")
        return parsed

    parsed: dict[str, Any] = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip().strip('"').strip("'")

    if not parsed:
        raise ValueError("Frontmatter parsed as empty mapping.")
    return parsed


def get_skill_version(path: str | Path) -> str:
    frontmatter = read_frontmatter(path)
    version = frontmatter.get("version")
    if version is None:
        raise ValueError(f"Skill file {path} missing frontmatter key: version")
    return str(version)


def hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
