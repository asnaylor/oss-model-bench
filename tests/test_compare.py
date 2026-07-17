from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from oss_model_bench.compare import compare_summaries
from oss_model_bench.util import write_json


class CompareTests(unittest.TestCase):
    def test_compares_common_performance_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.json"
            right = root / "right.json"
            write_json(left, {"kind": "performance", "run_id": "left", "status": "complete", "artifacts": [{"metrics": {"throughput": 100}}]})
            write_json(right, {"kind": "performance", "run_id": "right", "status": "complete", "artifacts": [{"metrics": {"throughput": 120}}]})
            result = compare_summaries(left, right)
        metric = result["metrics"]["[0].metrics.throughput"]
        self.assertEqual(metric["delta"], 20)
        self.assertEqual(metric["percent_change"], 20)


if __name__ == "__main__":
    unittest.main()
