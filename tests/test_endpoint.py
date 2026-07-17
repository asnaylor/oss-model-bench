from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from oss_model_bench.config import TargetConfig
from oss_model_bench.endpoint import check_target


class EndpointTests(unittest.TestCase):
    def test_check_passes_compatible_endpoint(self) -> None:
        target = TargetConfig(
            base_url="http://127.0.0.1:8000/v1",
            model="test-model",
            tokenizer="test-model",
            context_limit=8192,
            results_dir=Path("results"),
            api_key="test-key",
        )
        responses = [
            (200, b'{"data":[{"id":"test-model"}]}', {"Content-Type": "application/json"}),
            (200, b'{"choices":[{"message":{"content":"OMB_OK"}}]}', {"Content-Type": "application/json"}),
            (200, b'data: {"choices":[{"delta":{"content":"OMB"}}]}\n\ndata: [DONE]\n', {"Content-Type": "text/event-stream"}),
            (
                200,
                b'{"choices":[{"message":{"tool_calls":[{"function":{"name":"get_omb_status","arguments":"{}"}}]}}]}',
                {"Content-Type": "application/json"},
            ),
        ]
        with patch("oss_model_bench.endpoint._request", side_effect=responses) as request:
            result = check_target(target)
        self.assertEqual(result["status"], "passed")
        self.assertEqual({item["name"] for item in result["probes"]}, {"models", "chat", "streaming", "tool_calling"})
        self.assertEqual(request.call_count, 4)

    def test_tool_call_requires_structured_tool_output(self) -> None:
        target = TargetConfig("http://example.test/v1", "model", "model", 8192, Path("results"), "key")
        responses = [
            (200, b'{"data":[]}', {"Content-Type": "application/json"}),
            (200, b'{"choices":[{"message":{"content":"ok"}}]}', {"Content-Type": "application/json"}),
            (200, b'data: {}\n', {"Content-Type": "text/event-stream"}),
            (200, b'{"choices":[{"message":{"content":"I did it"}}]}', {"Content-Type": "application/json"}),
        ]
        with patch("oss_model_bench.endpoint._request", side_effect=responses):
            result = check_target(target)
        self.assertEqual(result["status"], "failed")
        tool_probe = next(item for item in result["probes"] if item["name"] == "tool_calling")
        self.assertIn("validation_error", tool_probe)


if __name__ == "__main__":
    unittest.main()
