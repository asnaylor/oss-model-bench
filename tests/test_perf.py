from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from oss_model_bench.config import TargetConfig
from oss_model_bench.perf import (
    build_agentic_profile_command,
    build_agentic_synthesis_command,
    build_baseline_commands,
    run_performance,
)
from oss_model_bench.util import read_json


class PerformanceTests(unittest.TestCase):
    def test_builds_six_baseline_profiles(self) -> None:
        target = self._target(Path("results"))
        commands = build_baseline_commands(target, Path("run"), duration=10)
        self.assertEqual(len(commands), 6)
        self.assertEqual({command[command.index("--concurrency") + 1] for command in commands}, {"1", "4", "16"})
        self.assertTrue(all("--no-server-metrics" in command for command in commands))

    def test_adds_optional_server_metrics_url_to_every_profile(self) -> None:
        target = self._target(Path("results"), server_metrics_url="http://vllm-node:8000/metrics")
        commands = build_baseline_commands(target, Path("run"), duration=10)
        self.assertTrue(all("--no-server-metrics" not in command for command in commands))
        self.assertTrue(
            all(command[command.index("--server-metrics") + 1] == target.server_metrics_url for command in commands)
        )
        agentic = build_agentic_profile_command(target, Path("run"), Path("trace/dataset.jsonl"), duration=10)
        self.assertEqual(agentic[agentic.index("--server-metrics") + 1], target.server_metrics_url)
        self.assertNotIn("--no-server-metrics", agentic)
        self.assertIn("--no-fixed-schedule", agentic)
        self.assertEqual(agentic[agentic.index("--concurrency") + 1], "1,4")

    def test_agentic_context_is_capped_at_200k(self) -> None:
        target = self._target(Path("results"), context_limit=300000)
        command = build_agentic_synthesis_command(target, Path("trace"))
        self.assertEqual(command[command.index("--max-isl") + 1], "200000")

    def test_dry_run_writes_redacted_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary_path = run_performance(self._target(Path(directory)), dry_run=True)
            text = summary_path.read_text()
            self.assertNotIn("top-secret", text)
            self.assertEqual(read_json(summary_path)["status"], "dry_run")

    @staticmethod
    def _target(
        results: Path,
        context_limit: int = 131072,
        server_metrics_url: str | None = None,
    ) -> TargetConfig:
        return TargetConfig(
            "https://example.test/v1",
            "model",
            "tokenizer",
            context_limit,
            results,
            "top-secret",
            server_metrics_url,
        )


if __name__ == "__main__":
    unittest.main()
