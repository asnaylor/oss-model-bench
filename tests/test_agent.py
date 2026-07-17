from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from oss_model_bench.agent import _opencode_config, build_bfcl_commands, load_panel, run_agent_panel
from oss_model_bench.agent_reports import collect_bfcl_report, collect_swe_generation, collect_swe_report
from oss_model_bench.config import TargetConfig
from oss_model_bench.util import read_json


PANEL = Path(__file__).parents[1] / "src" / "oss_model_bench" / "panels" / "panel-v1.json"


class AgentTests(unittest.TestCase):
    def test_panel_and_configs_are_endpoint_agnostic(self) -> None:
        panel = load_panel(PANEL)
        target = self._target(Path("results"))
        config = json.loads(_opencode_config(target))
        self.assertEqual(config["provider"]["omb"]["options"]["baseURL"], target.base_url)
        self.assertEqual(config["provider"]["omb"]["options"]["apiKey"], "{env:OMB_API_KEY}")
        self.assertNotIn("top-secret", json.dumps(config))
        self.assertEqual(len(panel["swe"]), 6)

    def test_bfcl_handler_alias_is_separate_from_served_target(self) -> None:
        panel = load_panel(PANEL)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OMB_BFCL_MODEL": "registered-handler"}):
            _, commands = build_bfcl_commands(self._target(Path(directory)), Path(directory) / "run", panel)
        self.assertIn("registered-handler", commands[0])

    def test_full_dry_run_needs_no_external_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary_path = run_agent_panel(self._target(Path(directory)), PANEL, dry_run=True)
            summary = read_json(summary_path)
        self.assertEqual(summary["status"], "dry_run")
        self.assertEqual(len(summary["tasks"]), 6)
        self.assertFalse(summary["official_comparable"])

    def test_collects_bfcl_partial_panel_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            score = run_dir / "native/bfcl/score/model/non_live/BFCL_v4_simple_python_score.json"
            score.parent.mkdir(parents=True)
            score.write_text(
                '{"accuracy":0.5,"correct_count":1,"total_count":2}\n'
                '{"id":"simple_python_1","valid":false}\n',
                encoding="utf-8",
            )
            report = collect_bfcl_report(run_dir, {"simple_python": ["simple_python_0", "simple_python_1"]})
        self.assertIsNotNone(report)
        self.assertEqual(report["correct_count"], 1)
        self.assertEqual(report["total_count"], 2)
        self.assertEqual(report["categories"][0]["category"], "simple_python")
        self.assertEqual(report["incorrect_ids"], ["simple_python_1"])
        self.assertTrue(report["complete_panel"])

    def test_collects_official_swe_report_and_generation_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "agent-run"
            run_dir.mkdir()
            patch_path = run_dir / "patches/task-a.patch"
            patch_path.parent.mkdir()
            patch_path.write_text("diff --git a/a b/a\n", encoding="utf-8")
            native = run_dir / "model.agent-run.json"
            native.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "total_instances": 2,
                        "completed_instances": 1,
                        "resolved_instances": 1,
                        "unresolved_instances": 0,
                        "empty_patch_instances": 1,
                        "error_instances": 0,
                        "resolved_ids": ["task-a"],
                        "unresolved_ids": [],
                        "error_ids": [],
                    }
                ),
                encoding="utf-8",
            )
            swe = collect_swe_report(run_dir, "model")
            generation = collect_swe_generation(
                run_dir,
                [
                    {"instance_id": "task-a", "status": "complete", "patch": "patches/task-a.patch", "opencode": {"returncode": 0, "duration_seconds": 5}},
                    {"instance_id": "task-b", "status": "failed", "patch": "patches/task-b.patch", "opencode": {"returncode": 124, "duration_seconds": 10}},
                ],
            )
        self.assertEqual(swe["resolution_rate"], 0.5)
        self.assertEqual(generation["patch_count"], 1)
        self.assertEqual(generation["empty_patch_ids"], ["task-b"])
        self.assertEqual(generation["timeout_ids"], ["task-b"])

    @staticmethod
    def _target(results: Path) -> TargetConfig:
        return TargetConfig("https://example.test/v1", "model", "tokenizer", 131072, results, "top-secret")


if __name__ == "__main__":
    unittest.main()
