"""Unit tests for post-extraction NLR feature analysis."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nlr_data.feature_analysis import (  # noqa: E402
    build_metadata_sensitivity,
    build_repeatability_summary,
    dfl_information_candidates,
    enrich_features,
    graybox_target_candidates,
)


def _base_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "workload_type": ["test"] * 6,
            "condition_id": ["a", "a", "b", "b", "c", "c"],
            "nodes": [1, 1, 2, 2, 3, 3],
            "target_utilization": [np.nan] * 6,
            "request_rate": [np.nan] * 6,
            "num_prompts": [np.nan] * 6,
            "batch_size": [np.nan] * 6,
            "max_output_tokens": [np.nan] * 6,
            "burstiness": [np.nan] * 6,
            "mean_power_w": [10, 10, 20, 20, 30, 30],
            "std_power_w": [1, 1, 2, 2, 3, 3],
            "min_power_w": [8, 8, 16, 16, 24, 24],
            "max_power_w": [12, 12, 24, 24, 36, 36],
            "peak_to_average_ratio": [1.2] * 6,
            "coefficient_of_variation": [0.1] * 6,
            "duration_s": [10] * 6,
            "energy_wh": np.array([10, 10, 20, 20, 30, 30]) * 10 / 3600,
            "mean_abs_ramp_w_per_s": [1, 1, 2, 2, 3, 3],
            "p95_abs_ramp_w_per_s": [2, 2, 4, 4, 6, 6],
            "p99_abs_ramp_w_per_s": [3, 3, 6, 6, 9, 9],
            "max_ramp_up_w_per_s": [4, 4, 8, 8, 12, 12],
            "max_ramp_down_w_per_s": [5, 5, 10, 10, 15, 15],
            "burst_total_duration_s": [2] * 6,
            "burst_mean_duration_s": [1] * 6,
            "autocorrelation_1s": [0.5] * 6,
            "autocorrelation_10s": [0.2] * 6,
            "dominant_period_s": [5] * 6,
        }
    )


class FeatureAnalysisTests(unittest.TestCase):
    def test_enrich_features_adds_normalised_quantities(self) -> None:
        enriched = enrich_features(_base_rows())
        self.assertAlmostEqual(float(enriched.loc[0, "power_range_pu"]), 0.4)
        self.assertAlmostEqual(float(enriched.loc[0, "p95_abs_ramp_pu_per_s"]), 0.2)
        self.assertAlmostEqual(float(enriched.loc[0, "burst_fraction"]), 0.2)
        self.assertAlmostEqual(float(enriched.loc[0, "energy_mean_relative_error"]), 0.0)

    def test_sensitivity_detects_monotonic_level_relation(self) -> None:
        enriched = enrich_features(_base_rows())
        result = build_metadata_sensitivity(enriched)
        row = result.loc[
            result["condition_variable"].eq("nodes")
            & result["feature"].eq("mean_power_w")
        ].iloc[0]
        self.assertAlmostEqual(float(row["spearman_rho"]), 1.0)

    def test_repeatability_uses_only_repeated_conditions(self) -> None:
        enriched = enrich_features(_base_rows())
        result = build_repeatability_summary(enriched)
        row = result.loc[result["feature"].eq("mean_power_w")].iloc[0]
        self.assertEqual(int(row["repeated_condition_count"]), 3)
        self.assertAlmostEqual(float(row["median_within_condition_cv"]), 0.0)

    def test_candidate_registries_are_nonempty_and_document_boundary(self) -> None:
        graybox = graybox_target_candidates()
        dfl = dfl_information_candidates()
        self.assertIn("p95_abs_ramp_pu_per_s", set(graybox["feature"]))
        self.assertTrue(dfl["system_boundary_note"].str.contains("PCC").all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
