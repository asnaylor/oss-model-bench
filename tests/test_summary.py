from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from oss_model_bench.summary import format_summary, latest_summary
from oss_model_bench.util import write_json


class SummaryTests(unittest.TestCase):
    def test_formats_key_performance_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "perf-run" / "summary.json"
            write_json(
                path,
                {
                    "kind": "performance",
                    "status": "complete",
                    "run_id": "perf-run",
                    "target": {"model": "model"},
                    "commands": [{"returncode": 0, "duration_seconds": 75}],
                    "artifacts": [
                        {
                            "path": "native/aiperf/baseline/chat-c4/profile_export_aiperf.json",
                            "metrics": {
                                "records.output_token_throughput.avg": 1234.5,
                                "records.request_throughput.avg": 4.2,
                                "records.ttft.avg": 30.1,
                                "records.ttft.p99": 80.2,
                                "records.itl.avg": 8.5,
                                "records.itl.p99": 12.0,
                                "records.request_latency.p99": 2000.0,
                                "records.request_count.avg": 250,
                                "records.error_count.avg": 0,
                            },
                        }
                    ],
                },
            )
            output = format_summary(path.parent)
        self.assertIn("OMB Performance", output)
        self.assertIn("chat-c4: output=1,234 tok/s", output)
        self.assertIn("TTFT=30.1/80.2 ms avg/p99", output)
        self.assertIn("ITL=8.50/12.0 ms avg/p99", output)
        self.assertIn("errors=0.00", output)

    def test_formats_official_agent_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            write_json(
                path,
                {
                    "kind": "agent_panel",
                    "status": "complete",
                    "bfcl_commands": [{"returncode": 0, "duration_seconds": 10}],
                    "bfcl_report": {
                        "correct_count": 8,
                        "total_count": 10,
                        "accuracy": 0.8,
                        "categories": [{"category": "simple_python", "correct_count": 8, "total_count": 10, "accuracy": 0.8}],
                        "incorrect_ids": ["simple_python_1", "simple_python_2"],
                    },
                    "tasks": [{"instance_id": "task-a", "status": "complete"}, {"instance_id": "task-b", "status": "failed"}],
                    "swe_generation": {"attempted": 2, "completed": 1, "patch_count": 1, "opencode_task_seconds": 30, "failed_ids": ["task-b"], "timeout_ids": [], "empty_patch_ids": ["task-b"]},
                    "swe_grade": {"returncode": 0, "duration_seconds": 20},
                    "swe_report": {
                        "total_instances": 2,
                        "completed_instances": 2,
                        "resolved_instances": 1,
                        "resolution_rate": 0.5,
                        "error_instances": 0,
                        "empty_patch_instances": 1,
                        "resolved_ids": ["task-a"],
                        "unresolved_ids": ["task-b"],
                        "error_ids": [],
                    },
                },
            )
            output = format_summary(path)
        self.assertIn("BFCL panel: 8/10 correct (80.0%", output)
        self.assertIn("SWE generation: 1/2 completed, 1 non-empty patches", output)
        self.assertIn("SWE-bench: 1/2 resolved (50.0%)", output)
        self.assertIn("Unresolved: task-b", output)

    def test_latest_summary_uses_modification_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = root / "perf-old" / "summary.json"
            newer = root / "agent-new" / "summary.json"
            write_json(older, {"kind": "performance"})
            write_json(newer, {"kind": "agent_panel"})
            os.utime(older, ns=(1, 1))
            os.utime(newer, ns=(2, 2))
            self.assertEqual(latest_summary(root), newer)


if __name__ == "__main__":
    unittest.main()
