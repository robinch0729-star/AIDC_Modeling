"""Unit tests for condition-dependent NLR residual generators."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlr_data.residual_models import (  # noqa: E402
    PowerProfile,
    ResidualLibrary,
    _ar1_phi,
    _simulate_ar1,
    feature_errors,
    simulate_residual,
    temporal_features,
)


class ResidualModelTests(unittest.TestCase):
    def test_ar1_estimation_and_generation_recover_target_scale(self) -> None:
        rng = np.random.default_rng(4)
        values = _simulate_ar1(100_000, phi=0.8, stationary_std=0.2, rng=rng)
        self.assertAlmostEqual(_ar1_phi(values), 0.8, delta=0.02)
        self.assertAlmostEqual(float(np.std(values)), 0.2, delta=0.01)

    def test_constant_baseline_has_finite_feature_error(self) -> None:
        time = np.arange(0.0, 180.0, 0.1)
        observed = 0.05 * np.sin(2 * np.pi * time / 5.0)
        errors = feature_errors(observed, np.zeros_like(observed), 0.1)
        self.assertTrue(np.isfinite(errors["overall_score"]))
        self.assertGreater(errors["overall_score"], 0.0)

    def test_white_noise_uses_condition_weighted_std(self) -> None:
        profile = PowerProfile(
            profile_id="donor",
            condition_id="donor",
            prompt_dataset="d",
            request_rate=10.0,
            split="train",
            time_s=np.arange(1000) * 0.1,
            power_kw=np.zeros(1000),
            interval_s=0.1,
            measurement_origin="synthetic",
        )
        statistics = pd.DataFrame(
            [
                {
                    "profile_id": "donor",
                    "prompt_dataset": "d",
                    "request_rate": 10.0,
                    "event_threshold_mad": threshold,
                    "residual_std_kw": 0.1,
                    "residual_ar1_phi": 0.8,
                    "background_std_kw": 0.08,
                    "background_ar1_phi": 0.7,
                    "burst_rate_per_s": 0.0,
                    "dip_rate_per_s": 0.0,
                }
                for threshold in (3.0, 4.0, 5.0)
            ]
        )
        library = ResidualLibrary(
            statistics=statistics,
            residuals={"donor": np.zeros(1000)},
            events=[],
            profiles={"donor": profile},
        )
        simulated = simulate_residual(
            "R1_white_noise",
            length=100_000,
            interval_s=0.1,
            prompt_dataset="d",
            request_rate=12.0,
            library=library,
            rng=np.random.default_rng(9),
        )
        self.assertAlmostEqual(float(np.std(simulated)), 0.1, delta=0.003)
        self.assertAlmostEqual(float(np.mean(simulated)), 0.0, delta=1e-12)

    def test_temporal_features_detect_ar1_correlation(self) -> None:
        values = _simulate_ar1(
            50_000,
            phi=0.9,
            stationary_std=0.1,
            rng=np.random.default_rng(11),
        )
        features = temporal_features(values, 0.1)
        self.assertGreater(features["acf_0p1s"], 0.85)
        self.assertGreater(features["std_kw"], 0.09)


if __name__ == "__main__":
    unittest.main()
