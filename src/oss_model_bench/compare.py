from __future__ import annotations

from pathlib import Path
from typing import Any

from .util import read_json, utc_now, write_json


def _numbers(value: Any, prefix: str = "") -> dict[str, float]:
    found: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            found.update(_numbers(child, name))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.update(_numbers(child, f"{prefix}[{index}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        found[prefix] = float(value)
    return found


def _agent_counts(summary: dict[str, Any]) -> dict[str, int]:
    tasks = summary.get("tasks", [])
    counts: dict[str, int] = {"total": len(tasks)}
    for task in tasks:
        status = str(task.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def compare_summaries(left_path: Path, right_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    left = read_json(left_path)
    right = read_json(right_path)
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError("summary files must contain JSON objects")
    if left.get("kind") != right.get("kind"):
        raise ValueError(f"cannot compare {left.get('kind')!r} with {right.get('kind')!r}")

    left_numbers = _numbers(left.get("artifacts", []))
    right_numbers = _numbers(right.get("artifacts", []))
    metrics: dict[str, dict[str, float | None]] = {}
    for name in sorted(left_numbers.keys() & right_numbers.keys()):
        before = left_numbers[name]
        after = right_numbers[name]
        metrics[name] = {
            "left": before,
            "right": after,
            "delta": after - before,
            "percent_change": ((after - before) / before * 100) if before else None,
        }

    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "comparison",
        "benchmark_kind": left.get("kind"),
        "created_at": utc_now(),
        "left": {"path": str(left_path), "run_id": left.get("run_id"), "status": left.get("status")},
        "right": {"path": str(right_path), "run_id": right.get("run_id"), "status": right.get("status")},
        "metrics": metrics,
    }
    if left.get("kind") == "agent_panel":
        result["task_status_counts"] = {"left": _agent_counts(left), "right": _agent_counts(right)}
        result["note"] = "Use the native BFCL and SWE-bench reports for official capability scores."
    if output_path:
        write_json(output_path, result)
    return result
