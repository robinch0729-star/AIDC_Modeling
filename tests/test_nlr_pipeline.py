"""Tests for manifest construction, resolution policy, and feature extraction."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nlr_data.feature_extraction import (  # noqa: E402
    build_condition_uncertainty,
    compute_profile_features,
)
from nlr_data.metadata_join import build_run_manifest  # noqa: E402
from nlr_data.readers import read_timeseries  # noqa: E402
from nlr_data.signal_validation import apply_analysis_policy  # noqa: E402


class NLRPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        default_root = PROJECT_ROOT.parent / "NLR_dataset"
        cls.data_root = Path(os.environ.get("NLR_DATA_ROOT", default_root)).resolve()
        if not cls.data_root.is_dir():
            raise unittest.SkipTest(f"NLR dataset not found: {cls.data_root}")
        cls.manifest = build_run_manifest(cls.data_root)

    def test_manifest_counts_and_exclusions(self) -> None:
        self.assertEqual(len(self.manifest), 2_475)
        self.assertEqual(int(self.manifest["eligible"].sum()), 2_473)
        excluded = self.manifest.loc[~self.manifest["eligible"]]
        self.assertEqual(len(excluded), 2)
        self.assertTrue(excluded["exclusion_reason"].eq("run_completed=False").all())
        self.assertTrue(self.manifest["path_exists"].all())
        self.assertFalse(self.manifest["relative_path"].duplicated().any())

    def test_online_rate_is_downsampled_to_point_one_seconds(self) -> None:
        record = self.manifest.loc[
            self.manifest["workload_type"].eq("online_inference_rate")
        ].iloc[0]
        raw = read_timeseries(
            self.data_root / record["relative_path"],
            dataset_root=self.data_root,
            max_rows=10_001,
            include_metadata_columns=False,
            validate=False,
        )
        analysis = apply_analysis_policy(raw, record)
        self.assertLess(len(analysis), len(raw) / 50)
        interval = analysis["time_s"].diff().dropna().median()
        self.assertAlmostEqual(float(interval), 0.1, places=9)

    def test_features_for_known_linear_signal(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": pd.Series(pd.NaT, index=range(3), dtype="datetime64[ns]"),
                "time_s": [0.0, 1.0, 2.0],
                "power_w": [0.0, 10.0, 20.0],
                "utilization": [np.nan, np.nan, np.nan],
            }
        )
        features = compute_profile_features(frame)
        self.assertAlmostEqual(features["mean_power_w"], 10.0)
        self.assertAlmostEqual(features["max_ramp_up_w_per_s"], 10.0)
        self.assertAlmostEqual(features["energy_wh"], 20.0 / 3600.0)

    def test_condition_uncertainty_uses_repeated_runs(self) -> None:
        features = pd.DataFrame(
            {
                "workload_type": ["test", "test"],
                "condition_id": ["same", "same"],
                "mean_power_w": [10.0, 14.0],
                "max_power_w": [15.0, 19.0],
                "std_power_w": [1.0, 2.0],
                "p95_abs_ramp_w_per_s": [3.0, 5.0],
                "max_ramp_up_w_per_s": [4.0, 6.0],
                "max_ramp_down_w_per_s": [2.0, 4.0],
                "energy_wh": [1.0, 1.5],
            }
        )
        summary = build_condition_uncertainty(features)
        self.assertEqual(int(summary.loc[0, "run_count"]), 2)
        self.assertAlmostEqual(
            float(summary.loc[0, "mean_power_w_condition_mean"]), 12.0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

