from __future__ import annotations

from pathlib import Path
from typing import Any

from .agent_reports import collect_bfcl_report, collect_swe_generation, collect_swe_report
from .util import read_json


def latest_summary(results_dir: Path) -> Path:
    candidates = [path for path in results_dir.glob("*/summary.json") if path.is_file()]
    candidates.extend(path for path in (results_dir / "check.json", results_dir / "summary.json") if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"no benchmark summaries found under {results_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def resolve_summary(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        direct = path / "summary.json"
        if direct.is_file():
            return direct
        check = path / "check.json"
        if check.is_file() and not any(path.glob("*/summary.json")):
            return check
        return latest_summary(path)
    raise FileNotFoundError(f"summary path does not exist: {path}")


def _number(metrics: dict[str, Any], *suffixes: str) -> float | None:
    for suffix in suffixes:
        matches = [
            (name, value)
            for name, value in metrics.items()
            if (name == suffix or name.endswith(f".{suffix}"))
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ]
        if matches:
            return float(min(matches, key=lambda item: len(item[0]))[1])
    return None


def _format_value(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _command_line(commands: list[dict[str, Any]]) -> str:
    if not commands:
        return "Commands: none recorded"
    succeeded = sum(command.get("returncode") == 0 for command in commands)
    elapsed = sum(float(command.get("duration_seconds", 0)) for command in commands)
    return f"Commands: {succeeded}/{len(commands)} succeeded, {elapsed:.1f}s total"


def _performance_lines(summary: dict[str, Any]) -> list[str]:
    commands = [item for item in summary.get("commands", []) if isinstance(item, dict)]
    lines = [_command_line(commands), "Profiles:"]
    artifacts = [item for item in summary.get("artifacts", []) if isinstance(item, dict)]
    if not artifacts:
        lines.append("  No AIPerf metric artifacts found")
        return lines

    for artifact in artifacts:
        path = Path(str(artifact.get("path", "unknown")))
        label = path.parent.name or path.name
        metrics = artifact.get("metrics", {})
        if not isinstance(metrics, dict):
            lines.append(f"  {label}: metrics unavailable")
            continue
        values: list[str] = []
        output_tps = _number(metrics, "output_token_throughput.avg")
        request_rate = _number(metrics, "request_throughput.avg")
        ttft_avg = _number(metrics, "ttft.avg")
        ttft_p99 = _number(metrics, "ttft.p99")
        itl_avg = _number(metrics, "itl.avg")
        itl_p99 = _number(metrics, "itl.p99")
        latency_p99 = _number(metrics, "request_latency.p99")
        request_count = _number(metrics, "request_count.avg", "request_count.count")
        error_count = _number(metrics, "error_count.avg", "error_count.count")
        error_rate = _number(metrics, "error_rate.avg")

        if output_tps is not None:
            values.append(f"output={_format_value(output_tps)} tok/s")
        if request_rate is not None:
            values.append(f"requests={_format_value(request_rate)}/s")
        if ttft_avg is not None or ttft_p99 is not None:
            ttft = "/".join(_format_value(value) if value is not None else "-" for value in (ttft_avg, ttft_p99))
            values.append(f"TTFT={ttft} ms avg/p99")
        if itl_avg is not None or itl_p99 is not None:
            itl = "/".join(_format_value(value) if value is not None else "-" for value in (itl_avg, itl_p99))
            values.append(f"ITL={itl} ms avg/p99")
        if latency_p99 is not None:
            values.append(f"latency-p99={_format_value(latency_p99)} ms")
        if request_count is not None:
            values.append(f"n={_format_value(request_count)}")
        if error_rate is not None:
            values.append(f"error-rate={_format_value(error_rate)}")
        elif error_count is not None:
            values.append(f"errors={_format_value(error_count)}")
        lines.append(f"  {label}: " + (" | ".join(values) if values else "no selected metrics found"))
    return lines


def _ids_line(label: str, values: Any) -> str | None:
    if not isinstance(values, list) or not values:
        return None
    ids = [str(value) for value in values]
    shown = ids[:10]
    suffix = f" (+{len(ids) - len(shown)} more)" if len(ids) > len(shown) else ""
    return f"  {label}: {', '.join(shown)}{suffix}"


def _percent(value: Any) -> str:
    return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else "n/a"


def _agent_lines(summary: dict[str, Any], run_dir: Path) -> list[str]:
    bfcl = [item for item in summary.get("bfcl_commands", []) if isinstance(item, dict)]
    tasks = [item for item in summary.get("tasks", []) if isinstance(item, dict)]
    lines = [_command_line(bfcl).replace("Commands:", "BFCL stages:", 1)]

    bfcl_report = summary.get("bfcl_report")
    if not isinstance(bfcl_report, dict):
        bfcl_report = collect_bfcl_report(run_dir)
    if isinstance(bfcl_report, dict) and int(bfcl_report.get("total_count", 0)) > 0:
        lines.append(
            f"BFCL panel: {bfcl_report['correct_count']}/{bfcl_report['total_count']} correct "
            f"({_percent(bfcl_report.get('accuracy'))}; partial, not leaderboard-comparable)"
        )
        for category in bfcl_report.get("categories", []):
            if isinstance(category, dict):
                lines.append(
                    f"  {category.get('category', 'unknown')}: "
                    f"{category.get('correct_count', 0)}/{category.get('total_count', 0)} "
                    f"({_percent(category.get('accuracy'))})"
                )
        incorrect = _ids_line("BFCL incorrect", bfcl_report.get("incorrect_ids"))
        if incorrect:
            lines.append(incorrect)
        if bfcl_report.get("parse_errors"):
            lines.append(f"  BFCL score parse errors: {len(bfcl_report['parse_errors'])}")
        if bfcl_report.get("count_mismatches"):
            mismatches = ", ".join(
                f"{name}={counts.get('actual', 0)}/{counts.get('expected', 0)}"
                for name, counts in sorted(bfcl_report["count_mismatches"].items())
            )
            lines.append(f"  BFCL incomplete categories: {mismatches}")
    else:
        lines.append("BFCL panel: score report unavailable")

    generation = summary.get("swe_generation")
    if not isinstance(generation, dict) and tasks:
        generation = collect_swe_generation(run_dir, tasks)
    if isinstance(generation, dict):
        lines.append(
            f"SWE generation: {generation.get('completed', 0)}/{generation.get('attempted', 0)} completed, "
            f"{generation.get('patch_count', 0)} non-empty patches, "
            f"{float(generation.get('opencode_task_seconds', 0)):.1f} OpenCode task-seconds"
        )
        for label, key in (("Generation failures", "failed_ids"), ("Generation timeouts", "timeout_ids"), ("Empty patches", "empty_patch_ids")):
            detail = _ids_line(label, generation.get(key))
            if detail:
                lines.append(detail)

    grade = summary.get("swe_grade")
    if isinstance(grade, dict):
        lines.append(
            f"SWE-bench grading: {'succeeded' if grade.get('returncode') == 0 else 'failed'}"
            f" (exit={grade.get('returncode')}, {float(grade.get('duration_seconds', 0)):.1f}s)"
        )
    else:
        lines.append("SWE-bench grading: not run")

    swe_report = summary.get("swe_report")
    model = summary.get("target", {}).get("model") if isinstance(summary.get("target"), dict) else None
    if not isinstance(swe_report, dict):
        swe_report = collect_swe_report(run_dir, str(model) if model else None)
    if isinstance(swe_report, dict):
        lines.append(
            f"SWE-bench: {swe_report.get('resolved_instances', 0)}/{swe_report.get('total_instances', 0)} resolved "
            f"({_percent(swe_report.get('resolution_rate'))}); completed={swe_report.get('completed_instances', 0)}, "
            f"errors={swe_report.get('error_instances', 0)}, empty={swe_report.get('empty_patch_instances', 0)}"
        )
        for label, key in (("Resolved", "resolved_ids"), ("Unresolved", "unresolved_ids"), ("Grading errors", "error_ids")):
            detail = _ids_line(label, swe_report.get(key))
            if detail:
                lines.append(detail)
    else:
        lines.append("SWE-bench result: official report unavailable")
    for error in summary.get("report_errors", []):
        lines.append(f"Report error: {error}")
    return lines


def _check_lines(summary: dict[str, Any]) -> list[str]:
    lines = ["Probes:"]
    for probe in summary.get("probes", []):
        if not isinstance(probe, dict):
            continue
        status = "PASS" if probe.get("ok") else "FAIL"
        details = [f"HTTP {probe['status_code']}"] if "status_code" in probe else []
        if "duration_seconds" in probe:
            details.append(f"{float(probe['duration_seconds']):.2f}s")
        if probe.get("validation_error"):
            details.append(str(probe["validation_error"]))
        lines.append(f"  {probe.get('name', 'unknown')}: {status}" + (f" ({'; '.join(details)})" if details else ""))
    return lines


def format_summary(path: Path) -> str:
    resolved = resolve_summary(path)
    summary = read_json(resolved)
    if not isinstance(summary, dict):
        raise ValueError(f"summary must contain a JSON object: {resolved}")
    kind = str(summary.get("kind", "unknown"))
    title = {
        "performance": "Performance",
        "agent_panel": "Agent capability",
        "target_check": "Target check",
    }.get(kind, kind.replace("_", " ").title())
    target = summary.get("target", {})
    model = target.get("model") if isinstance(target, dict) else None
    lines = [
        f"OMB {title}",
        f"Status: {str(summary.get('status', 'unknown')).upper()}",
    ]
    if summary.get("run_id"):
        lines.append(f"Run: {summary['run_id']}")
    if model:
        lines.append(f"Model: {model}")
    if kind == "performance":
        lines.extend(_performance_lines(summary))
    elif kind == "agent_panel":
        lines.extend(_agent_lines(summary, resolved.parent))
    elif kind == "target_check":
        lines.extend(_check_lines(summary))
    else:
        lines.append("No specialized formatter is available for this result kind")
    lines.append(f"Source: {resolved}")
    return "\n".join(lines)
