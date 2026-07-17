from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from oss_model_bench.agent import _opencode_config, build_bfcl_commands, load_panel, run_agent_panel
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

    @staticmethod
    def _target(results: Path) -> TargetConfig:
        return TargetConfig("https://example.test/v1", "model", "tokenizer", 131072, results, "top-secret")


if __name__ == "__main__":
    unittest.main()
