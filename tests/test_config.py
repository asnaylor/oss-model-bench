from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from oss_model_bench.config import ConfigError, TargetConfig, normalize_base_url, normalize_server_metrics_url


class ConfigTests(unittest.TestCase):
    def test_loads_target_and_never_exposes_key(self) -> None:
        values = {
            "OMB_BASE_URL": "https://example.test/v1/",
            "OMB_MODEL": "org/model",
            "OMB_API_KEY": "secret-value",
            "OMB_CONTEXT_LIMIT": "200000",
            "OMB_RESULTS_DIR": "/tmp/omb-results",
            "OMB_SERVER_METRICS_URL": "http://vllm-node:8000/metrics",
        }
        with patch.dict(os.environ, values, clear=True):
            target = TargetConfig.from_env()
        self.assertEqual(target.base_url, "https://example.test/v1")
        self.assertEqual(target.tokenizer, "org/model")
        self.assertEqual(target.context_limit, 200000)
        self.assertEqual(target.results_dir, Path("/tmp/omb-results"))
        self.assertEqual(target.server_metrics_url, "http://vllm-node:8000/metrics")
        self.assertNotIn("api_key", target.public_dict())

    def test_rejects_embedded_credentials(self) -> None:
        with self.assertRaises(ConfigError):
            normalize_base_url("https://user:password@example.test/v1")

    def test_rejects_invalid_server_metrics_url(self) -> None:
        with self.assertRaises(ConfigError):
            normalize_server_metrics_url("vllm-node:8000/metrics")

    def test_reports_missing_values(self) -> None:
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ConfigError, "OMB_BASE_URL"):
            TargetConfig.from_env()


if __name__ == "__main__":
    unittest.main()
