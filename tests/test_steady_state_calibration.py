"""Tests for the NLR request-rate steady-state gray-box workflow."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibration.steady_state_calibration import fit_steady_state_model  # noqa: E402
from modeling.request_rate_splits import (  # noqa: E402
    attach_split,
    build_request_rate_splits,
)
from models.steady_state_models import MODEL_SPECS  # noqa: E402
from nlr_data.condition_targets import build_online_rate_targets  # noqa: E402


class SteadyStateCalibrationTests(unittest.TestCase):
    def test_all_candidate_models_are_monotone_and_saturating(self) -> None:
        request_rate = np.linspace(0.0, 5000.0, 1001)
        example_parameters = {
            "exponential": np.array([2.0, 1.0, 20.0]),
            "hill": np.array([2.0, 1.0, 20.0, 2.0]),
            "piecewise_capped_linear": np.array([2.0, 0.05, 20.0]),
        }
        for name, spec in MODEL_SPECS.items():
            prediction = spec.predict(request_rate, example_parameters[name])
            self.assertTrue(np.all(np.diff(prediction) >= -1e-12), name)
            self.assertLess(abs(prediction[-1] - 3.0), 1e-3, name)

    def test_grouped_rate_split_has_no_cross_dataset_leakage(self) -> None:
        targets = pd.DataFrame(
            {
                "request_rate": np.tile(np.arange(1, 21), 2),
                "prompt_dataset": np.repeat(["a", "b"], 20),
            }
        )
        assignments = build_request_rate_splits(targets)
        interpolation = attach_split(targets, assignments, "interpolation")
        per_rate_partitions = interpolation.groupby("request_rate")["partition"].nunique()
        self.assertTrue((per_rate_partitions == 1).all())
        self.assertEqual(set(interpolation["partition"]), {"train", "test"})

    def test_exponential_fit_recovers_synthetic_curve(self) -> None:
        request_rate = np.linspace(1.0, 100.0, 100)
        truth = np.array([2.05, 0.85, 18.0])
        target = MODEL_SPECS["exponential"].predict(request_rate, truth)
        fit = fit_steady_state_model("exponential", request_rate, target)
        self.assertTrue(fit.success)
        np.testing.assert_allclose(fit.parameters, truth, rtol=1e-4, atol=1e-4)

    def test_target_builder_keeps_prompt_dataset_as_a_factor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            features = pd.DataFrame(
                {
                    "profile_id": ["a", "b"],
                    "relative_path": ["a.csv", "b.csv"],
                    "workload_type": ["online_inference_rate"] * 2,
                    "measurement_scope": ["job_total_IT_power"] * 2,
                    "measurement_origin": ["interpolated_from_measured"] * 2,
                    "request_rate": [1.0, 1.0],
                    "mean_power_w": [2000.0, 2100.0],
                    "std_power_w": [10.0, 20.0],
                    "duration_s": [180.0, 180.0],
                    "p05_power_w": [1980.0, 2060.0],
                    "p95_power_w": [2020.0, 2140.0],
                }
            )
            audit = pd.DataFrame(
                {
                    "relative_path": ["a.csv", "b.csv"],
                    "dataset_path": ["org/dataset-a", "org/dataset-b"],
                    "recommended_condition_id": ["a|1", "b|1"],
                    "analysis_cohort": ["measured_it_workload"] * 2,
                    "eligible": [True, True],
                }
            )
            feature_path = directory / "features.csv"
            audit_path = directory / "audit.csv"
            features.to_csv(feature_path, index=False)
            audit.to_csv(audit_path, index=False)
            result = build_online_rate_targets(feature_path, audit_path)
            self.assertEqual(set(result["prompt_dataset"]), {"dataset-a", "dataset-b"})
            self.assertEqual(result["recommended_condition_id"].nunique(), 2)
            self.assertTrue(
                result["uncertainty_interpretation"].str.contains("not run-to-run").all()
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

