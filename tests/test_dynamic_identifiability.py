"""Regression tests for the NLR dynamic-identifiability audit."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlr_data.dynamic_identifiability import (  # noqa: E402
    M3Parameters,
    analyze_profile,
)


class DynamicIdentifiabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = M3Parameters(p0_kw=2.1, delta_p_kw=0.7, k=4.0, x0=3.0)
        self.record = {
            "profile_id": "synthetic",
            "relative_path": "synthetic.csv",
            "prompt_dataset": "fixture",
            "request_rate": 25.0,
            "measurement_origin": "synthetic",
        }

    def test_m3_prediction_is_monotone_and_bounded(self) -> None:
        rate = np.linspace(0.0, 500.0, 1000)
        prediction = self.parameters.predict(rate)
        self.assertTrue(np.all(np.diff(prediction) >= 0.0))
        self.assertGreaterEqual(prediction.min(), self.parameters.p0_kw)
        self.assertLessEqual(
            prediction.max(), self.parameters.saturation_power_kw + 1e-12
        )

    def test_sustained_profile_never_claims_causal_tau(self) -> None:
        time_s = np.arange(0.0, 180.0, 0.1)
        steady = float(self.parameters.predict(self.record["request_rate"]))
        rng = np.random.default_rng(7)
        power_kw = steady + 0.02 * rng.standard_normal(len(time_s))
        frame = pd.DataFrame({"time_s": time_s, "power_w": power_kw * 1000.0})

        identifiability, residual, _, _ = analyze_profile(
            frame, self.record, self.parameters
        )

        self.assertFalse(identifiability["input_transition_signal_available"])
        self.assertFalse(identifiability["causal_tau_identifiable"])
        self.assertEqual(
            identifiability["tau_identifiability_status"],
            "not_estimable_no_input_transition",
        )
        self.assertAlmostEqual(residual["m3_bias_kw"], 0.0, delta=0.003)
        self.assertGreater(residual["residual_std_kw"], 0.0)

    def test_output_start_fit_is_only_apparent_even_for_exponential_trace(self) -> None:
        time_s = np.arange(0.0, 180.0, 0.1)
        steady = float(self.parameters.predict(self.record["request_rate"]))
        power_kw = steady - 0.5 * np.exp(-time_s / 4.0)
        frame = pd.DataFrame({"time_s": time_s, "power_w": power_kw * 1000.0})

        identifiability, _, _, _ = analyze_profile(
            frame, self.record, self.parameters
        )

        self.assertTrue(identifiability["startup_transient_candidate"])
        # The production fit uses a robust median over the first second rather
        # than the exact t=0 sample, so verify recovery to the correct scale.
        self.assertGreater(identifiability["apparent_tau_fit_s"], 3.0)
        self.assertLess(identifiability["apparent_tau_fit_s"], 6.0)
        self.assertFalse(identifiability["causal_tau_identifiable"])
        self.assertEqual(
            identifiability["tau_identifiability_status"],
            "apparent_output_transient_only_no_input_signal",
        )


if __name__ == "__main__":
    unittest.main()
