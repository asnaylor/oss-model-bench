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
            (
                200,
                b'data: {"choices":[{"delta":{"content":"OMB"}}]}\n\ndata: {"choices":[{"delta":{"content":"_OK"}}]}\n\ndata: [DONE]\n',
                {"Content-Type": "text/event-stream"},
            ),
            (
                200,
                b'{"choices":[{"finish_reason":"tool_calls","message":{"tool_calls":[{"function":{"name":"get_omb_status","arguments":"{\\"component\\":\\"endpoint\\"}"}}]}}]}',
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
            (200, b'{"choices":[{"message":{"content":"OMB_OK"}}]}', {"Content-Type": "application/json"}),
            (200, b'data: {"choices":[{"delta":{"content":"OMB_OK"}}]}\n\ndata: [DONE]\n', {"Content-Type": "text/event-stream"}),
            (200, b'{"choices":[{"message":{"content":"I did it"}}]}', {"Content-Type": "application/json"}),
        ]
        with patch("oss_model_bench.endpoint._request", side_effect=responses):
            result = check_target(target)
        self.assertEqual(result["status"], "failed")
        tool_probe = next(item for item in result["probes"] if item["name"] == "tool_calling")
        self.assertIn("validation_error", tool_probe)

    def test_reasoning_only_length_response_fails_chat_contract(self) -> None:
        target = TargetConfig("http://example.test/v1", "model", "model", 8192, Path("results"), "key")
        responses = [
            (200, b'{"data":[]}', {"Content-Type": "application/json"}),
            (
                200,
                b'{"choices":[{"finish_reason":"length","message":{"content":null,"reasoning_content":"thinking"}}]}',
                {"Content-Type": "application/json"},
            ),
            (200, b'data: {"choices":[{"delta":{"content":"OMB_OK"}}]}\n\ndata: [DONE]\n', {"Content-Type": "text/event-stream"}),
        ]
        with patch("oss_model_bench.endpoint._request", side_effect=responses):
            result = check_target(target, include_tools=False)
        self.assertEqual(result["status"], "failed")
        chat_probe = next(item for item in result["probes"] if item["name"] == "chat")
        self.assertIn("validation_error", chat_probe)

    def test_check_uses_reasoning_safe_completion_budgets(self) -> None:
        target = TargetConfig("http://example.test/v1", "model", "model", 8192, Path("results"), "key")
        responses = [
            (200, b'{"data":[]}', {"Content-Type": "application/json"}),
            (200, b'{"choices":[{"message":{"content":"OMB_OK"}}]}', {"Content-Type": "application/json"}),
            (200, b'data: {"choices":[{"delta":{"content":"OMB_OK"}}]}\n\ndata: [DONE]\n', {"Content-Type": "text/event-stream"}),
            (
                200,
                b'{"choices":[{"message":{"tool_calls":[{"function":{"name":"get_omb_status","arguments":{"component":"endpoint"}}}]}}]}',
                {"Content-Type": "application/json"},
            ),
        ]
        with patch("oss_model_bench.endpoint._request", side_effect=responses) as request:
            check_target(target)
        self.assertEqual(request.call_args_list[1].kwargs["payload"]["max_tokens"], 256)
        self.assertEqual(request.call_args_list[2].kwargs["payload"]["max_tokens"], 256)
        self.assertEqual(request.call_args_list[3].kwargs["payload"]["max_tokens"], 512)


if __name__ == "__main__":
    unittest.main()
