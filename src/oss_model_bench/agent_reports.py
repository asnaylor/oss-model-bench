from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .util import read_json


def _json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _bfcl_category(path: Path) -> str:
    stem = path.stem.removesuffix("_score")
    parts = stem.split("_", 2)
    return parts[2] if len(parts) == 3 and parts[0] == "BFCL" else stem


def collect_bfcl_report(run_dir: Path, expected_cases: dict[str, Any] | None = None) -> dict[str, Any] | None:
    score_root = run_dir / "native" / "bfcl" / "score"
    score_files = sorted(score_root.rglob("*_score.json"))
    if not score_files:
        return None

    categories: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    incorrect_ids: list[str] = []
    for path in score_files:
        try:
            records = _json_records(path)
            header = records[0] if records else {}
            correct = int(header["correct_count"])
            total = int(header["total_count"])
            if total < 0 or correct < 0 or correct > total:
                raise ValueError("invalid correct/total counts")
            category_incorrect = sorted(
                str(record["id"])
                for record in records[1:]
                if isinstance(record.get("id"), str) and not record.get("valid", False)
            )
            incorrect_ids.extend(category_incorrect)
            categories.append(
                {
                    "category": _bfcl_category(path),
                    "accuracy": (correct / total) if total else None,
                    "correct_count": correct,
                    "total_count": total,
                    "incorrect_ids": category_incorrect,
                    "path": str(path.relative_to(run_dir)),
                }
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            parse_errors.append({"path": str(path.relative_to(run_dir)), "error": str(exc)})

    correct = sum(category["correct_count"] for category in categories)
    total = sum(category["total_count"] for category in categories)
    report = {
        "partial_panel": True,
        "accuracy": (correct / total) if total else None,
        "correct_count": correct,
        "total_count": total,
        "incorrect_ids": sorted(incorrect_ids),
        "categories": categories,
        "parse_errors": parse_errors,
    }
    if expected_cases is not None:
        expected_counts = {
            str(category): len(case_ids)
            for category, case_ids in expected_cases.items()
            if isinstance(case_ids, list)
        }
        actual_counts = {category["category"]: category["total_count"] for category in categories}
        count_mismatches = {
            category: {"expected": expected, "actual": actual_counts.get(category, 0)}
            for category, expected in expected_counts.items()
            if actual_counts.get(category, 0) != expected
        }
        report["expected_total_count"] = sum(expected_counts.values())
        report["count_mismatches"] = count_mismatches
        report["complete_panel"] = not parse_errors and not count_mismatches and total == sum(expected_counts.values())
    return report


def collect_swe_report(run_dir: Path, model: str | None = None) -> dict[str, Any] | None:
    candidates: list[Path] = []
    if model:
        candidates.append(run_dir / f"{model.replace('/', '__')}.{run_dir.name}.json")
    candidates.extend(sorted(run_dir.glob(f"*.{run_dir.name}.json")))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            report = read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(report, dict) or not {
            "total_instances",
            "resolved_instances",
            "resolved_ids",
            "unresolved_ids",
        }.issubset(report):
            continue
        result = dict(report)
        total = int(result.get("total_instances", 0))
        resolved = int(result.get("resolved_instances", 0))
        result["resolution_rate"] = (resolved / total) if total else None
        result["path"] = str(path.relative_to(run_dir))
        return result
    return None


def collect_swe_generation(run_dir: Path, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(task.get("status", "unknown")) for task in tasks)
    empty_patch_ids: list[str] = []
    patch_ids: list[str] = []
    timeout_ids: list[str] = []
    failed_ids: list[str] = []
    task_seconds = 0.0
    for task in tasks:
        instance_id = str(task.get("instance_id", "unknown"))
        patch_path = run_dir / str(task.get("patch", ""))
        if patch_path.is_file() and patch_path.stat().st_size > 0:
            patch_ids.append(instance_id)
        else:
            empty_patch_ids.append(instance_id)
        opencode = task.get("opencode")
        if isinstance(opencode, dict):
            task_seconds += float(opencode.get("duration_seconds", 0))
            if opencode.get("returncode") == 124:
                timeout_ids.append(instance_id)
        if task.get("status") != "complete":
            failed_ids.append(instance_id)
    return {
        "attempted": len(tasks),
        "completed": statuses.get("complete", 0),
        "statuses": dict(sorted(statuses.items())),
        "patch_count": len(patch_ids),
        "patch_ids": sorted(patch_ids),
        "empty_patch_count": len(empty_patch_ids),
        "empty_patch_ids": sorted(empty_patch_ids),
        "timeout_ids": sorted(timeout_ids),
        "failed_ids": sorted(failed_ids),
        "opencode_task_seconds": round(task_seconds, 3),
    }
