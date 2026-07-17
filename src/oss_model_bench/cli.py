from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .agent import run_agent_panel
from .compare import compare_summaries
from .config import ConfigError, TargetConfig
from .endpoint import check_target
from .perf import run_performance
from .util import write_json


def _default_panel() -> Path:
    return Path(__file__).resolve().parent / "panels" / "panel-v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omb",
        description="Fast performance and agent/SWE benchmarks for OpenAI-compatible endpoints.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="probe chat, streaming, and tool-call compatibility")
    check.add_argument("--no-tools", action="store_true", help="skip the tool-calling probe")
    check.add_argument("--output", type=Path, help="write the check JSON to this path")

    perf = subparsers.add_parser("perf", help="run AIPerf baseline and agentic-context load tests")
    perf.add_argument("--phase", choices=("all", "baseline", "agentic"), default="all")
    perf.add_argument("--baseline-duration", type=int, default=60, metavar="SECONDS")
    perf.add_argument("--agentic-duration", type=int, default=300, metavar="SECONDS")
    perf.add_argument("--dry-run", action="store_true")

    agent = subparsers.add_parser("agent", help="run BFCL and OpenCode/SWE-bench capability tests")
    agent.add_argument("--panel", type=Path, default=_default_panel())
    agent.add_argument("--workers", type=int, default=4)
    agent.add_argument("--task-timeout", type=int, default=1500, metavar="SECONDS")
    agent.add_argument("--no-bfcl", action="store_true")
    agent.add_argument("--no-swe", action="store_true")
    agent.add_argument("--no-grade", action="store_true")
    agent.add_argument("--dry-run", action="store_true")

    compare = subparsers.add_parser("compare", help="compare two benchmark summary files")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--output", type=Path)
    return parser


def _positive(parser: argparse.ArgumentParser, name: str, value: int) -> int:
    if value <= 0:
        parser.error(f"{name} must be greater than zero")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            result = compare_summaries(args.left, args.right, args.output)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        target = TargetConfig.from_env()
        if args.command == "check":
            result = check_target(target, include_tools=not args.no_tools)
            output = args.output or target.results_dir / "check.json"
            write_json(output, result)
            print(output)
            return 0 if result["status"] == "passed" else 1

        if args.command == "perf":
            summary = run_performance(
                target,
                phase=args.phase,
                dry_run=args.dry_run,
                baseline_duration=_positive(parser, "--baseline-duration", args.baseline_duration),
                agentic_duration=_positive(parser, "--agentic-duration", args.agentic_duration),
            )
            print(summary)
            return 0

        if args.command == "agent":
            if args.no_bfcl and args.no_swe:
                parser.error("--no-bfcl and --no-swe would leave no capability tests to run")
            summary = run_agent_panel(
                target,
                args.panel,
                workers=_positive(parser, "--workers", args.workers),
                task_timeout=_positive(parser, "--task-timeout", args.task_timeout),
                run_bfcl=not args.no_bfcl,
                run_swe=not args.no_swe,
                grade=not args.no_grade,
                dry_run=args.dry_run,
            )
            print(summary)
            return 0
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"omb: error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
