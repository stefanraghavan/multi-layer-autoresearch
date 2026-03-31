"""Agent execution: run_agent via OpenAI Responses API with local shell + skills."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

from openai import AsyncOpenAI

from lib.config import AgentExecution, MODEL_ID
from lib.skill_file import read_frontmatter
from lib.utils import _log, _to_int

# Per-token pricing (USD per token) — used as fallback when SDK doesn't report cost.
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.4": (2.50 / 1_000_000, 10.00 / 1_000_000),
    "gpt-5.4-thinking": (5.00 / 1_000_000, 20.00 / 1_000_000),
}

_MAX_AGENT_TURNS = 200

_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI()
    return _client


def _estimate_cost(model: str, tokens_in: Optional[int], tokens_out: Optional[int]) -> Optional[float]:
    rates = _MODEL_PRICING.get(model)
    if rates is None:
        return None
    in_cost = (tokens_in or 0) * rates[0]
    out_cost = (tokens_out or 0) * rates[1]
    total = in_cost + out_cost
    return total if total > 0 else None


def _discover_skills(runtime_dir: Path) -> list[dict[str, str]]:
    """Find all skill directories under .claude/skills/ and build LocalSkill dicts."""
    skills_root = runtime_dir / ".claude" / "skills"
    if not skills_root.exists():
        return []

    skills: list[dict[str, str]] = []
    for skill_dir in sorted(skills_root.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        try:
            fm = read_frontmatter(skill_md)
            name = str(fm.get("name", skill_dir.name))
            description = str(fm.get("description", f"Skill: {name}"))
        except Exception:
            name = skill_dir.name
            description = f"Skill: {name}"

        skills.append({
            "name": name,
            "description": description,
            "path": str(skill_dir),
        })
    return skills


def _build_tools(runtime_dir: Path) -> list[dict[str, Any]]:
    """Build the tools array for the OpenAI Responses API with local shell + skills."""
    skills = _discover_skills(runtime_dir)
    shell_tool: dict[str, Any] = {
        "type": "shell",
        "environment": {
            "type": "local",
            "skills": skills,
        },
    }
    return [shell_tool]


def _execute_local_shell(action: Any, cwd: str) -> str:
    """Execute a shell_call or local_shell_call action and return output as JSON."""
    commands = getattr(action, "commands", None)
    if commands:
        command = ["bash", "-c", " && ".join(commands)]
    else:
        command = getattr(action, "command", None) or []
        if isinstance(command, str):
            command = [command]
        else:
            command = list(command)

    working_dir = getattr(action, "working_directory", None) or cwd
    timeout_ms = getattr(action, "timeout_ms", None)
    timeout_s = (timeout_ms / 1000) if timeout_ms else 600  # default 10 min

    try:
        import os
        run_env = dict(os.environ)
        env_vars = getattr(action, "env", None)
        if env_vars and isinstance(env_vars, dict):
            run_env.update(env_vars)

        result = subprocess.run(
            command,
            cwd=working_dir,
            env=run_env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

        output = {
            "stdout": result.stdout[-50000:] if len(result.stdout) > 50000 else result.stdout,
            "stderr": result.stderr[-10000:] if len(result.stderr) > 10000 else result.stderr,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        output = {"stdout": "", "stderr": "Command timed out", "exit_code": -1}
    except Exception as exc:
        output = {"stdout": "", "stderr": str(exc), "exit_code": -1}

    return json.dumps(output)


def _serialize_output_item(item: Any) -> dict[str, Any]:
    """Serialize a response output item for replay, stripping read-only fields."""
    if hasattr(item, "model_dump"):
        d = item.model_dump(exclude_none=True)
    else:
        d = dict(item)
    for key in ("created_by",):
        d.pop(key, None)
    return d


async def run_agent(
    runtime_dir: Path,
    prompt: str,
    allowed_tools: list[str],
) -> AgentExecution:
    client = _get_client()
    tools = _build_tools(runtime_dir)
    cwd = str(runtime_dir)

    instructions = (
        f"You are working in: {runtime_dir}\n"
        f"Always operate within this directory. Read the CLAUDE.md file for project context."
    )

    started = perf_counter()
    num_skills = len(tools[0]["environment"]["skills"])
    _log(f"  -> calling OpenAI responses.create (model={MODEL_ID}, skills={num_skills})")

    total_in = 0
    total_out = 0
    turn_count = 0

    conversation: list[dict[str, Any]] = []
    conversation.append({"type": "message", "role": "user", "content": prompt})

    response = await client.responses.create(
        model=MODEL_ID,
        input=prompt,
        instructions=instructions,
        tools=tools,
        reasoning={"effort": "xhigh"},
    )

    while turn_count < _MAX_AGENT_TURNS:
        turn_count += 1

        usage = getattr(response, "usage", None)
        if usage is not None:
            total_in += _to_int(getattr(usage, "input_tokens", None)) or 0
            total_out += _to_int(getattr(usage, "output_tokens", None)) or 0

        output_items = getattr(response, "output", []) or []
        output_types = [getattr(item, "type", "?") for item in output_items]
        _log(f"  <- turn {turn_count} output types: {output_types}")

        shell_calls = [
            item for item in output_items
            if getattr(item, "type", None) in ("local_shell_call", "shell_call")
        ]

        if not shell_calls:
            break

        _log(f"  -> turn {turn_count}: executing {len(shell_calls)} shell call(s)")

        shell_outputs: list[dict[str, Any]] = []
        for call in shell_calls:
            call_id = getattr(call, "call_id", None) or getattr(call, "id", "")
            call_type = getattr(call, "type", "shell_call")
            action = getattr(call, "action", None)

            if action is not None:
                if call_type == "shell_call":
                    cmds = getattr(action, "commands", [])
                    cmd_str = "; ".join(cmds) if isinstance(cmds, (list, tuple)) else str(cmds)
                else:
                    cmd = getattr(action, "command", [])
                    cmd_str = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
                _log(f"     $ {cmd_str[:120]}")

                shell_result = await asyncio.to_thread(
                    _execute_local_shell, action, cwd
                )
                parsed = json.loads(shell_result)
            else:
                parsed = {"stdout": "", "stderr": "No action", "exit_code": -1}

            if call_type == "shell_call":
                shell_outputs.append({
                    "type": "shell_call_output",
                    "call_id": call_id,
                    "output": [{
                        "stdout": parsed.get("stdout", ""),
                        "stderr": parsed.get("stderr", ""),
                        "outcome": {
                            "type": "exit",
                            "exit_code": parsed.get("exit_code", -1),
                        },
                    }],
                })
            else:
                shell_outputs.append({
                    "type": "local_shell_call_output",
                    "id": call_id,
                    "output": shell_result,
                })

        conversation.extend(
            _serialize_output_item(item) for item in output_items
        )
        conversation.extend(shell_outputs)

        response = await client.responses.create(
            model=MODEL_ID,
            input=conversation,
            instructions=instructions,
            tools=tools,
            reasoning={"effort": "xhigh"},
        )

    result_text = response.output_text or ""

    _log(f"  <- agent complete: {turn_count} turns, tokens_in={total_in}, tokens_out={total_out}")
    if result_text:
        _log(f"  <- result preview: {result_text[:200]}...")

    cost_usd = _estimate_cost(MODEL_ID, total_in or None, total_out or None)

    return AgentExecution(
        result_text=result_text,
        duration_ms=int((perf_counter() - started) * 1000),
        tokens_in=total_in or None,
        tokens_out=total_out or None,
        cost_usd=cost_usd,
    )
