from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .config import TargetConfig
from .util import utc_now


def _request(
    target: TargetConfig,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> tuple[int, bytes, dict[str, str]]:
    headers = {"Accept": "application/json"}
    data = None
    if target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(target.endpoint(path), data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers.items())


def _probe(name: str, operation: Any) -> dict[str, Any]:
    started = time.monotonic()
    try:
        status, body, headers = operation()
        decoded = body.decode("utf-8", errors="replace")
        parsed: Any
        try:
            parsed = json.loads(decoded)
        except json.JSONDecodeError:
            parsed = None
        return {
            "name": name,
            "ok": 200 <= status < 300,
            "status_code": status,
            "duration_seconds": round(time.monotonic() - started, 3),
            "content_type": headers.get("Content-Type", ""),
            "response": parsed if parsed is not None else decoded[:500],
        }
    except Exception as exc:  # noqa: BLE001 - probes report errors instead of aborting
        return {
            "name": name,
            "ok": False,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _has_chat_choice(response: Any) -> bool:
    return (
        isinstance(response, dict)
        and isinstance(response.get("choices"), list)
        and bool(response["choices"])
        and isinstance(response["choices"][0], dict)
    )


def _validate_probe(probe: dict[str, Any]) -> None:
    if not probe.get("ok"):
        return
    response = probe.get("response")
    name = probe["name"]
    valid = True
    if name == "chat":
        valid = _has_chat_choice(response)
    elif name == "streaming":
        content_type = str(probe.get("content_type", "")).lower()
        valid = "text/event-stream" in content_type or (isinstance(response, str) and "data:" in response)
    elif name == "tool_calling":
        valid = _has_chat_choice(response)
        if valid:
            message = response["choices"][0].get("message", {})
            valid = isinstance(message, dict) and bool(message.get("tool_calls"))
    if not valid:
        probe["ok"] = False
        probe["validation_error"] = f"response did not satisfy the {name} compatibility contract"


def check_target(target: TargetConfig, *, include_tools: bool = True) -> dict[str, Any]:
    completion = {
        "model": target.model,
        "messages": [{"role": "user", "content": "Reply with exactly OMB_OK"}],
        "temperature": 0,
        "max_tokens": 16,
    }
    probes = [
        _probe("models", lambda: _request(target, "models", timeout=15)),
        _probe("chat", lambda: _request(target, "chat/completions", payload=completion)),
        _probe(
            "streaming",
            lambda: _request(target, "chat/completions", payload={**completion, "stream": True}),
        ),
    ]
    if include_tools:
        tool_payload = {
            "model": target.model,
            "messages": [{"role": "user", "content": "Call get_omb_status with component='endpoint'."}],
            "temperature": 0,
            "max_tokens": 64,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_omb_status",
                        "description": "Return benchmark component status.",
                        "parameters": {
                            "type": "object",
                            "properties": {"component": {"type": "string"}},
                            "required": ["component"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "get_omb_status"}},
        }
        probes.append(_probe("tool_calling", lambda: _request(target, "chat/completions", payload=tool_payload)))

    for probe in probes:
        _validate_probe(probe)

    required = {"chat", "streaming"}
    if include_tools:
        required.add("tool_calling")
    required_ok = all(probe["ok"] for probe in probes if probe["name"] in required)
    return {
        "schema_version": 1,
        "kind": "target_check",
        "created_at": utc_now(),
        "status": "passed" if required_ok else "failed",
        "target": target.public_dict(),
        "probes": probes,
    }
