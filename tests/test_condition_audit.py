"""Integration tests for NLR experiment-condition auditing."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nlr_data.condition_audit import (  # noqa: E402
    build_audited_manifest,
    build_collision_report,
    build_validation_summary,
)


class ConditionAuditIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        default_root = PROJECT_ROOT.parent / "NLR_dataset"
        cls.data_root = Path(os.environ.get("NLR_DATA_ROOT", default_root)).resolve()
        if not cls.data_root.is_dir():
            raise unittest.SkipTest(f"NLR dataset not found: {cls.data_root}")
        cls.audited = build_audited_manifest(cls.data_root)
        cls.summary = build_validation_summary(cls.audited).set_index("workload_type")

    def test_offline_condition_count_is_preserved(self) -> None:
        row = self.summary.loc["offline_inference"]
        self.assertEqual(int(row["current_condition_count"]), 80)
        self.assertEqual(int(row["recommended_condition_count"]), 80)

    def test_online_conditions_include_prompt_dataset(self) -> None:
        finite = self.summary.loc["online_inference_finite"]
        rate = self.summary.loc["online_inference_rate"]
        self.assertEqual(int(finite["current_condition_count"]), 57)
        self.assertEqual(int(finite["recommended_condition_count"]), 114)
        self.assertEqual(int(rate["current_condition_count"]), 100)
        self.assertEqual(int(rate["recommended_condition_count"]), 200)

    def test_collision_report_is_limited_to_online_groups(self) -> None:
        collisions = build_collision_report(self.audited)
        self.assertEqual(
            set(collisions["workload_type"]),
            {"online_inference_finite", "online_inference_rate"},
        )
        self.assertEqual(len(collisions), 157)


if __name__ == "__main__":
    unittest.main(verbosity=2)
