from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from oss_model_bench.util import run_command


class UtilTests(unittest.TestCase):
    def test_timeout_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_command(
                [sys.executable, "-c", "import time; time.sleep(2)"],
                stderr_path=Path(directory) / "stderr.log",
                timeout=0.05,
            )
        self.assertEqual(result.returncode, 124)

    def test_command_is_redacted_in_result(self) -> None:
        result = run_command(["program", "secret"], dry_run=True, secrets=("secret",))
        self.assertEqual(result.command, ["program", "<redacted>"])

    def test_announces_redacted_command_and_completion(self) -> None:
        output = io.StringIO()
        with redirect_stderr(output):
            result = run_command(
                [sys.executable, "-c", "pass", "top-secret"],
                secrets=("top-secret",),
                announce="test profile",
            )
        self.assertEqual(result.returncode, 0)
        progress = output.getvalue()
        self.assertIn("[omb] running test profile:", progress)
        self.assertIn("[omb] completed test profile: exit=0", progress)
        self.assertIn("<redacted>", progress)
        self.assertNotIn("top-secret", progress)


if __name__ == "__main__":
    unittest.main()
