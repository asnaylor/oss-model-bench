from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import TargetConfig
from .util import require_command, run_command, run_id, utc_now, write_json

BASELINE_WORKLOADS = (
    {"name": "chat", "isl": 1024, "osl": 256},
    {"name": "code", "isl": 4096, "osl": 512},
)
CONCURRENCIES = (1, 4, 16)


def _aiperf_url(base_url: str) -> str:
    return base_url[:-3] if base_url.endswith("/v1") else base_url


def _common_args(target: TargetConfig, artifact_dir: Path) -> list[str]:
    command = [
        "aiperf",
        "profile",
        "--model",
        target.model,
        "--tokenizer",
        target.tokenizer,
        "--url",
        _aiperf_url(target.base_url),
        "--endpoint-type",
        "chat",
        "--streaming",
        "--artifact-dir",
        str(artifact_dir),
        "--no-server-metrics",
    ]
    if target.api_key:
        command.extend(["--api-key", target.api_key])
    return command


def build_baseline_commands(target: TargetConfig, run_dir: Path, *, duration: int = 60) -> list[list[str]]:
    commands: list[list[str]] = []
    for workload in BASELINE_WORKLOADS:
        for concurrency in CONCURRENCIES:
            artifact_dir = run_dir / "native" / "aiperf" / "baseline" / f"{workload['name']}-c{concurrency}"
            commands.append(
                _common_args(target, artifact_dir)
                + [
                    "--isl",
                    str(workload["isl"]),
                    "--osl",
                    str(workload["osl"]),
                    "--concurrency",
                    str(concurrency),
                    "--warmup-duration",
                    "15",
                    "--benchmark-duration",
                    str(duration),
                    "--benchmark-grace-period",
                    "30",
                    "--random-seed",
                    "42",
                ]
            )
    return commands


def build_agentic_synthesis_command(target: TargetConfig, trace_dir: Path) -> list[str]:
    target_context = min(200_000, int(target.context_limit * 0.8))
    return [
        "aiperf",
        "synthesize",
        "agentic-code",
        "--num-sessions",
        "4",
        "--seed",
        "42",
        "--max-isl",
        str(target_context),
        "--max-osl",
        "256",
        "--output",
        str(trace_dir),
    ]


def build_agentic_profile_command(
    target: TargetConfig,
    run_dir: Path,
    dataset_path: Path,
    *,
    duration: int = 300,
) -> list[str]:
    artifact_dir = run_dir / "native" / "aiperf" / "agentic"
    return _common_args(target, artifact_dir) + [
        "--input-file",
        str(dataset_path),
        "--custom-dataset-type",
        "mooncake_trace",
        "--concurrency",
        "1,4",
        "--warmup-duration",
        "15",
        "--benchmark-duration",
        str(duration),
        "--benchmark-grace-period",
        "60",
        "--random-seed",
        "42",
    ]


def _extract_numbers(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_extract_numbers(child, next_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        lowered = prefix.lower()
        interesting = ("throughput", "latency", "ttft", "itl", "error", "request_count")
        if any(token in lowered for token in interesting):
            result[prefix] = float(value)
    return result


def collect_aiperf_artifacts(run_dir: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("native/aiperf/**/profile_export_aiperf.json")):
        entry: dict[str, Any] = {"path": str(path.relative_to(run_dir))}
        try:
            native = json.loads(path.read_text(encoding="utf-8"))
            entry["metrics"] = _extract_numbers(native)
        except (OSError, json.JSONDecodeError) as exc:
            entry["parse_error"] = str(exc)
        artifacts.append(entry)
    return artifacts


def run_performance(
    target: TargetConfig,
    *,
    phase: str = "all",
    dry_run: bool = False,
    baseline_duration: int = 60,
    agentic_duration: int = 300,
) -> Path:
    if not dry_run:
        require_command("aiperf")
    identifier = run_id("perf")
    run_dir = target.results_dir / identifier
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    command_results: list[dict[str, Any]] = []
    env = os.environ.copy()

    if phase in {"all", "baseline"}:
        for index, command in enumerate(build_baseline_commands(target, run_dir, duration=baseline_duration)):
            result = run_command(
                command,
                env=env,
                stdout_path=run_dir / "logs" / f"baseline-{index}.out",
                stderr_path=run_dir / "logs" / f"baseline-{index}.err",
                timeout=baseline_duration + 180,
                dry_run=dry_run,
                secrets=(target.api_key,),
            )
            command_results.append(result.__dict__)
            if result.returncode != 0:
                break

    if phase in {"all", "agentic"} and all(item["returncode"] == 0 for item in command_results):
        trace_dir = run_dir / "native" / "aiperf" / "agentic-trace"
        synth = run_command(
            build_agentic_synthesis_command(target, trace_dir),
            env=env,
            stdout_path=run_dir / "logs" / "agentic-synthesize.out",
            stderr_path=run_dir / "logs" / "agentic-synthesize.err",
            timeout=120,
            dry_run=dry_run,
            secrets=(target.api_key,),
        )
        command_results.append(synth.__dict__)
        if synth.returncode == 0:
            datasets = sorted(trace_dir.glob("**/dataset.jsonl")) if not dry_run else []
            dataset_path = datasets[-1] if datasets else trace_dir / "<generated>" / "dataset.jsonl"
            profile = run_command(
                build_agentic_profile_command(target, run_dir, dataset_path, duration=agentic_duration),
                env=env,
                stdout_path=run_dir / "logs" / "agentic-profile.out",
                stderr_path=run_dir / "logs" / "agentic-profile.err",
                timeout=(2 * agentic_duration) + 300,
                dry_run=dry_run,
                secrets=(target.api_key,),
            )
            command_results.append(profile.__dict__)

    failed = [item for item in command_results if item["returncode"] != 0]
    summary = {
        "schema_version": 1,
        "kind": "performance",
        "run_id": identifier,
        "created_at": started_at,
        "finished_at": utc_now(),
        "status": "dry_run" if dry_run else ("failed" if failed else "complete"),
        "phase": phase,
        "target": target.public_dict(),
        "commands": command_results,
        "artifacts": [] if dry_run else collect_aiperf_artifacts(run_dir),
    }
    summary_path = run_dir / "summary.json"
    write_json(summary_path, summary)
    return summary_path
