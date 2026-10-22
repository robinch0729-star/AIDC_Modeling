"""Tests for portable workload generation and facility aggregation."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlr_data.dynamic_identifiability import M3Parameters  # noqa: E402
from nlr_data.residual_models import (  # noqa: E402
    EventTemplate,
    PowerProfile,
    ResidualLibrary,
)
from nlr_data.workload_generator import WorkloadGenerator  # noqa: E402


def synthetic_generator(model: str = "R1_white_noise") -> WorkloadGenerator:
    interval = 0.1
    residual = np.sin(np.arange(1800) * interval * 2 * np.pi / 5.0) * 0.1
    profile = PowerProfile(
        profile_id="donor",
        condition_id="donor",
        prompt_dataset="dataset",
        request_rate=50.0,
        split="train",
        time_s=np.arange(len(residual)) * interval,
        power_kw=2.5 + residual,
        interval_s=interval,
        measurement_origin="synthetic",
    )
    statistics = pd.DataFrame(
        [
            {
                "profile_id": "donor",
                "condition_id": "donor",
                "prompt_dataset": "dataset",
                "request_rate": 50.0,
                "split": "train",
                "interval_s": interval,
                "m3_steady_power_kw": 2.5,
                "profile_mean_power_kw": 2.5,
                "m3_bias_kw": 0.0,
                "residual_std_kw": 0.1,
                "residual_ar1_phi": 0.9,
                "residual_correlation_time_s": 1.0,
                "event_threshold_mad": threshold,
                "background_std_kw": 0.08,
                "background_ar1_phi": 0.85,
                "burst_count": 1,
                "dip_count": 1,
                "burst_rate_per_s": 0.01,
                "dip_rate_per_s": 0.01,
                "burst_mean_duration_s": 0.2,
                "dip_mean_duration_s": 0.2,
                "burst_mean_peak_kw": 0.3,
                "dip_mean_peak_kw": -0.3,
                "event_fraction": 0.01,
            }
            for threshold in (3.0, 4.0, 5.0)
        ]
    )
    events = [
        EventTemplate(
            profile_id="donor",
            prompt_dataset="dataset",
            request_rate=50.0,
            threshold_mad=3.0,
            sign=sign,
            waveform_kw=np.asarray(waveform),
            duration_s=0.3,
            peak_kw=float(waveform[1]),
        )
        for sign, waveform in (
            ("burst", [0.1, 0.3, 0.1]),
            ("dip", [-0.1, -0.3, -0.1]),
        )
    ]
    library = ResidualLibrary(
        statistics=statistics,
        residuals={"donor": residual},
        events=events,
        profiles={"donor": profile},
    )
    return WorkloadGenerator(
        m3_parameters={
            "dataset": M3Parameters(p0_kw=2.0, delta_p_kw=0.5, k=4.0, x0=3.0)
        },
        residual_library=library,
        residual_model=model,
        artifact_metadata={"fixture": True},
    )


class WorkloadGeneratorTests(unittest.TestCase):
    def test_facility_profile_preserves_requested_mean(self) -> None:
        generator = synthetic_generator()
        result = generator.generate_facility_profile(
            prompt_dataset="dataset",
            request_rate=50.0,
            target_it_mean_mw=20.0,
            n_effective=10,
            common_mode_fraction=0.1,
            seed=2,
        )
        self.assertAlmostEqual(float(result["it_power_mw"].mean()), 20.0, places=10)
        self.assertTrue((result["it_power_mw"] >= 0.0).all())

    def test_independent_diversity_reduces_variation(self) -> None:
        generator = synthetic_generator()
        one = generator.generate_facility_profile(
            prompt_dataset="dataset",
            request_rate=50.0,
            target_it_mean_mw=20.0,
            n_effective=1,
            common_mode_fraction=0.0,
            seed=8,
        )
        many = generator.generate_facility_profile(
            prompt_dataset="dataset",
            request_rate=50.0,
            target_it_mean_mw=20.0,
            n_effective=100,
            common_mode_fraction=0.0,
            seed=8,
        )
        self.assertLess(float(many["it_power_mw"].std()), float(one["it_power_mw"].std()) / 5.0)

    def test_fully_common_profile_does_not_depend_on_effective_unit_count(self) -> None:
        generator = synthetic_generator()
        one = generator.generate_facility_profile(
            prompt_dataset="dataset",
            request_rate=50.0,
            target_it_mean_mw=20.0,
            n_effective=1,
            common_mode_fraction=1.0,
            seed=14,
        )
        many = generator.generate_facility_profile(
            prompt_dataset="dataset",
            request_rate=50.0,
            target_it_mean_mw=20.0,
            n_effective=100,
            common_mode_fraction=1.0,
            seed=14,
        )
        np.testing.assert_allclose(one["it_power_mw"], many["it_power_mw"])

    def test_artifact_round_trip_is_reproducible(self) -> None:
        generator = synthetic_generator("R3_ar1_event_mad3")
        with tempfile.TemporaryDirectory() as temporary:
            path = generator.save_artifact(Path(temporary) / "artifact")
            loaded = WorkloadGenerator.from_artifact(path)
            original = generator.generate_unit_profile(
                prompt_dataset="dataset", request_rate=50.0, seed=21
            )
            restored = loaded.generate_unit_profile(
                prompt_dataset="dataset", request_rate=50.0, seed=21
            )
            np.testing.assert_allclose(original["power_kw"], restored["power_kw"])
            self.assertEqual(loaded.residual_model, generator.residual_model)


if __name__ == "__main__":
    unittest.main()
