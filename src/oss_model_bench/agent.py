from __future__ import annotations

import concurrent.futures
import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import TargetConfig
from .util import read_json, require_command, run_command, run_id, utc_now, write_json

DEFAULT_SWE_DATASET = "princeton-nlp/SWE-bench_Lite"


def load_panel(path: Path) -> dict[str, Any]:
    panel = read_json(path)
    if not isinstance(panel, dict) or panel.get("schema_version") != 1:
        raise ValueError(f"unsupported panel format: {path}")
    if not isinstance(panel.get("bfcl"), dict) or not isinstance(panel.get("swe"), list):
        raise ValueError(f"panel must contain bfcl and swe sections: {path}")
    return panel


def build_bfcl_commands(target: TargetConfig, run_dir: Path, panel: dict[str, Any]) -> tuple[dict[str, str], list[list[str]]]:
    bfcl_root = run_dir / "native" / "bfcl"
    bfcl_root.mkdir(parents=True, exist_ok=True)
    write_json(bfcl_root / "test_case_ids_to_generate.json", panel["bfcl"])
    env = os.environ.copy()
    env.update(
        {
            "BFCL_PROJECT_ROOT": str(bfcl_root),
            "REMOTE_OPENAI_BASE_URL": target.base_url,
            "REMOTE_OPENAI_API_KEY": target.api_key or "",
            "REMOTE_OPENAI_TOKENIZER_PATH": target.tokenizer,
        }
    )
    result_dir = bfcl_root / "result"
    score_dir = bfcl_root / "score"
    bfcl_model = os.getenv("OMB_BFCL_MODEL", target.model)
    generate = [
        "bfcl",
        "generate",
        "--model",
        bfcl_model,
        "--run-ids",
        "--skip-server-setup",
        "--result-dir",
        str(result_dir),
        "--num-threads",
        "4",
    ]
    evaluate = [
        "bfcl",
        "evaluate",
        "--model",
        bfcl_model,
        "--partial-eval",
        "--result-dir",
        str(result_dir),
        "--score-dir",
        str(score_dir),
    ]
    return env, [generate, evaluate]


def _opencode_config(target: TargetConfig) -> str:
    return json.dumps(
        {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "omb": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "oss-model-bench",
                    "options": {"baseURL": target.base_url, "apiKey": "{env:OMB_API_KEY}"},
                    "models": {
                        target.model: {
                            "name": target.model,
                            "limit": {"context": target.context_limit, "output": min(65536, target.context_limit // 4)},
                        }
                    },
                }
            },
            "permission": {"*": "allow"},
        },
        separators=(",", ":"),
    )


def _load_swe_rows(panel: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("SWE tasks require: uv sync --extra agent") from exc

    cache: dict[tuple[str, str, str], Any] = {}
    rows: list[dict[str, Any]] = []
    for selector in panel["swe"]:
        key = (selector["dataset"], selector["revision"], selector["split"])
        if key not in cache:
            cache[key] = load_dataset(key[0], split=key[2], revision=key[1])
        row = dict(cache[key][int(selector["index"])])
        row["dataset"] = selector["dataset"]
        row["split"] = selector["split"]
        rows.append(row)
    return rows


def _prepare_repo(row: dict[str, Any], task_dir: Path, *, dry_run: bool, api_key: str | None) -> list[dict[str, Any]]:
    repo = str(row["repo"])
    logs = task_dir.parent.parent / "logs" / str(row.get("instance_id", repo.replace("/", "__")))
    commands = [
        ["git", "clone", "--quiet", "--no-checkout", f"https://github.com/{repo}.git", str(task_dir)],
        ["git", "-C", str(task_dir), "checkout", "--quiet", str(row["base_commit"])],
    ]
    results = []
    for index, command in enumerate(commands):
        result = run_command(
            command,
            stdout_path=logs.with_name(logs.name + f"-prepare-{index}.out"),
            stderr_path=logs.with_name(logs.name + f"-prepare-{index}.err"),
            dry_run=dry_run,
            secrets=(api_key,),
        )
        results.append(result.__dict__)
        if result.returncode != 0:
            break
    return results


def _run_opencode_task(
    target: TargetConfig,
    row: dict[str, Any],
    run_dir: Path,
    *,
    timeout: int,
    dry_run: bool,
) -> dict[str, Any]:
    instance_id = str(row.get("instance_id", f"row-{row['repo']}"))
    task_dir = run_dir / "work" / instance_id
    task_result_path = run_dir / "tasks" / f"{instance_id}.json"
    if task_result_path.exists() and not dry_run:
        existing = read_json(task_result_path)
        if isinstance(existing, dict) and existing.get("status") == "complete":
            return existing

    prepare = _prepare_repo(row, task_dir, dry_run=dry_run, api_key=target.api_key)
    if any(item["returncode"] != 0 for item in prepare):
        patch_path = run_dir / "patches" / f"{instance_id}.patch"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.touch()
        result = {
            "instance_id": instance_id,
            "dataset": row.get("dataset", DEFAULT_SWE_DATASET),
            "repo": row["repo"],
            "status": "prepare_failed",
            "commands": prepare,
            "patch": str(patch_path.relative_to(run_dir)),
        }
        write_json(task_result_path, result)
        return result

    prompt = (
        "Resolve the following repository issue. Inspect the code, make the smallest correct change, "
        "and run relevant tests when possible. Do not merely describe a patch; edit the working tree. "
        "Do not commit the changes.\n\n"
        f"{row['problem_statement']}"
    )
    env = os.environ.copy()
    env.update(
        {
            "OMB_API_KEY": target.api_key or "",
            "OPENCODE_CONFIG_CONTENT": _opencode_config(target),
            "OPENCODE_DISABLE_AUTOUPDATE": "true",
            "OPENCODE_DISABLE_DEFAULT_PLUGINS": "true",
            "OPENCODE_DISABLE_LSP_DOWNLOAD": "true",
            "OPENCODE_DISABLE_MODELS_FETCH": "true",
            "OPENCODE_AUTO_SHARE": "false",
        }
    )
    logs = run_dir / "logs" / instance_id
    command = [
        "opencode",
        "--pure",
        "run",
        "--model",
        f"omb/{target.model}",
        "--format",
        "json",
        "--auto",
        "--dir",
        str(task_dir),
        prompt,
    ]
    execution = run_command(
        command,
        env=env,
        stdout_path=logs.with_name(logs.name + "-opencode.jsonl"),
        stderr_path=logs.with_name(logs.name + "-opencode.err"),
        timeout=timeout,
        dry_run=dry_run,
        secrets=(target.api_key,),
    )
    intent = run_command(
        ["git", "-C", str(task_dir), "add", "--intent-to-add", "--all"],
        stdout_path=logs.with_name(logs.name + "-git-intent.out"),
        stderr_path=logs.with_name(logs.name + "-git-intent.err"),
        dry_run=dry_run,
        secrets=(target.api_key,),
    )
    patch_path = run_dir / "patches" / f"{instance_id}.patch"
    patch = run_command(
        ["git", "-C", str(task_dir), "diff", "--binary", str(row["base_commit"])],
        stdout_path=patch_path,
        stderr_path=logs.with_name(logs.name + "-git-diff.err"),
        dry_run=dry_run,
        secrets=(target.api_key,),
    )
    result = {
        "instance_id": instance_id,
        "dataset": row.get("dataset", DEFAULT_SWE_DATASET),
        "repo": row["repo"],
        "status": "dry_run"
        if dry_run
        else ("complete" if execution.returncode == 0 and intent.returncode == 0 and patch.returncode == 0 else "failed"),
        "opencode": execution.__dict__,
        "git_intent": intent.__dict__,
        "patch": str(patch_path.relative_to(run_dir)),
    }
    write_json(task_result_path, result)
    return result


def _write_predictions(run_dir: Path, target: TargetConfig, tasks: list[dict[str, Any]]) -> Path:
    path = run_dir / "predictions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for task in tasks:
        patch_path = run_dir / str(task["patch"])
        patch = patch_path.read_text(encoding="utf-8") if patch_path.exists() else ""
        lines.append(
            json.dumps(
                {
                    "instance_id": task["instance_id"],
                    "model_name_or_path": target.model,
                    "model_patch": patch,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def _grade_swe(
    run_dir: Path,
    predictions: Path,
    task_ids: list[str],
    *,
    dataset: str,
    split: str,
    dry_run: bool,
    api_key: str | None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset,
        "--split",
        split,
        "--predictions_path",
        str(predictions),
        "--max_workers",
        "4",
        "--run_id",
        run_dir.name,
        "--instance_ids",
        *task_ids,
    ]
    result = run_command(
        command,
        cwd=run_dir,
        stdout_path=run_dir / "logs" / "swe-grade.out",
        stderr_path=run_dir / "logs" / "swe-grade.err",
        dry_run=dry_run,
        secrets=(api_key,),
    )
    return result.__dict__


def run_agent_panel(
    target: TargetConfig,
    panel_path: Path,
    *,
    workers: int = 4,
    task_timeout: int = 1500,
    run_bfcl: bool = True,
    run_swe: bool = True,
    grade: bool = True,
    dry_run: bool = False,
) -> Path:
    panel = load_panel(panel_path)
    identifier = run_id("agent")
    run_dir = target.results_dir / identifier
    run_dir.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = []

    if run_bfcl:
        if not dry_run:
            require_command("bfcl")
        bfcl_env, bfcl_commands = build_bfcl_commands(target, run_dir, panel)
        for index, command in enumerate(bfcl_commands):
            result = run_command(
                command,
                env=bfcl_env,
                stdout_path=run_dir / "logs" / f"bfcl-{index}.out",
                stderr_path=run_dir / "logs" / f"bfcl-{index}.err",
                timeout=1200 if index == 0 else 300,
                dry_run=dry_run,
                secrets=(target.api_key,),
            )
            commands.append(result.__dict__)
            if result.returncode != 0:
                break

    task_results: list[dict[str, Any]] = []
    grade_result: dict[str, Any] | None = None
    if run_swe:
        if not dry_run:
            require_command("opencode")
            require_command("git")
        rows = _load_swe_rows(panel) if not dry_run else [
            {
                "instance_id": f"panel-row-{item['index']}",
                "repo": "<resolved-from-dataset>",
                "base_commit": "<resolved-from-dataset>",
                "problem_statement": "<resolved-from-dataset>",
            }
            for item in panel["swe"]
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _run_opencode_task,
                    target,
                    row,
                    run_dir,
                    timeout=task_timeout,
                    dry_run=dry_run,
                )
                for row in rows
            ]
            for future, row in zip(futures, rows, strict=True):
                try:
                    task_results.append(future.result())
                except Exception as exc:  # noqa: BLE001 - preserve other task results
                    instance_id = str(row.get("instance_id", "unknown"))
                    patch_path = run_dir / "patches" / f"{instance_id}.patch"
                    patch_path.parent.mkdir(parents=True, exist_ok=True)
                    patch_path.touch()
                    task_results.append(
                        {
                            "instance_id": instance_id,
                            "dataset": row.get("dataset", DEFAULT_SWE_DATASET),
                            "repo": row.get("repo", "unknown"),
                            "status": "runner_error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "patch": str(patch_path.relative_to(run_dir)),
                        }
                    )
        predictions = _write_predictions(run_dir, target, task_results)
        if grade:
            dataset_splits = {(str(row.get("dataset", DEFAULT_SWE_DATASET)), str(row.get("split", "dev"))) for row in rows}
            if len(dataset_splits) != 1:
                raise ValueError("one agent run can grade only one SWE dataset and split")
            dataset, split = dataset_splits.pop()
            grade_result = _grade_swe(
                run_dir,
                predictions,
                [str(row["instance_id"]) for row in rows],
                dataset=dataset,
                split=split,
                dry_run=dry_run,
                api_key=target.api_key,
            )

    failed = [item for item in commands if item["returncode"] != 0]
    failed.extend(item for item in task_results if item["status"] not in {"complete", "dry_run"})
    if grade_result and grade_result["returncode"] != 0:
        failed.append(grade_result)
    summary = {
        "schema_version": 1,
        "kind": "agent_panel",
        "run_id": identifier,
        "created_at": utc_now(),
        "finished_at": utc_now(),
        "status": "dry_run" if dry_run else ("failed" if failed else "complete"),
        "target": target.public_dict(),
        "panel": {"name": panel["name"], "path": str(panel_path)},
        "bfcl_commands": commands,
        "tasks": task_results,
        "swe_grade": grade_result,
        "official_comparable": False,
    }
    summary_path = run_dir / "summary.json"
    write_json(summary_path, summary)
    return summary_path
