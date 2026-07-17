from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from oss_model_bench.config import TargetConfig
from oss_model_bench.perf import build_agentic_synthesis_command, build_baseline_commands, run_performance
from oss_model_bench.util import read_json


class PerformanceTests(unittest.TestCase):
    def test_builds_six_baseline_profiles(self) -> None:
        target = self._target(Path("results"))
        commands = build_baseline_commands(target, Path("run"), duration=10)
        self.assertEqual(len(commands), 6)
        self.assertEqual({command[command.index("--concurrency") + 1] for command in commands}, {"1", "4", "16"})

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
    def _target(results: Path, context_limit: int = 131072) -> TargetConfig:
        return TargetConfig("https://example.test/v1", "model", "tokenizer", context_limit, results, "top-secret")


if __name__ == "__main__":
    unittest.main()
