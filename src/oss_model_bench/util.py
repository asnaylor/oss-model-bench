from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def run_id(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temp, 0o600)
    temp.replace(path)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def require_command(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise RuntimeError(f"required command not found: {name}")
    return executable


def redact(text: str, secrets: Iterable[str | None]) -> str:
    result = text
    for secret in secrets:
        if secret:
            result = result.replace(secret, "<redacted>")
    return result


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    duration_seconds: float
    stdout_path: str | None
    stderr_path: str | None


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    timeout: float | None = None,
    dry_run: bool = False,
    secrets: Iterable[str | None] = (),
    announce: str | None = None,
) -> CommandResult:
    safe_command = [redact(part, secrets) for part in command]
    if dry_run:
        print(shlex.join(safe_command))
        return CommandResult(safe_command, 0, 0.0, None, None)

    if announce:
        print(f"[omb] running {announce}: {shlex.join(safe_command)}", file=sys.stderr, flush=True)

    if stdout_path:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if stderr_path:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = stdout_path.open("w", encoding="utf-8") if stdout_path else subprocess.PIPE
    stderr_handle = stderr_path.open("w", encoding="utf-8") if stderr_path else subprocess.PIPE
    started = time.monotonic()
    try:
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                text=True,
                stdout=stdout_handle,
                stderr=stderr_handle,
                timeout=timeout,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
            if stderr_path:
                stderr_handle.write(f"command timed out after {timeout} seconds\n")
    finally:
        if stdout_path:
            stdout_handle.close()
        if stderr_path:
            stderr_handle.close()
    duration_seconds = round(time.monotonic() - started, 3)
    if announce:
        outcome = "completed" if returncode == 0 else "failed"
        print(
            f"[omb] {outcome} {announce}: exit={returncode} elapsed={duration_seconds:.1f}s",
            file=sys.stderr,
            flush=True,
        )
    return CommandResult(
        safe_command,
        returncode,
        duration_seconds,
        str(stdout_path) if stdout_path else None,
        str(stderr_path) if stderr_path else None,
    )
