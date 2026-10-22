"""Small, deterministic tests of the new causal/stateful connection."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aidc_facility.time_varying_it import (
    FixedService, R3Coefficients, advance_residual, coefficients_at,
    make_dispatch, profile_metrics, reference_design_scale,
    simulate_case, validate_dispatch,
)
from nlr_data.final_it_profile_generator import FinalITProfileGenerator
from test_final_it_profile_generator import _artifact


class TimeVaryingITTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.generator = FinalITProfileGenerator.from_benchmark_artifact(_artifact(Path(self.temp.name)))
        self.services = (FixedService("s", "d", 4, 15.0, 30.0),)
        self.traffic = pd.DataFrame({"time_s": [0.0, 1.0], "bin_s": [1.0, 1.0], "count": [10, 20]})

    def tearDown(self):
        self.temp.cleanup()

    def dispatch(self, variable=True, traffic=None):
        return make_dispatch(self.traffic if traffic is None else traffic, self.services, {"s": "count"}, 1.0, variable=variable)

    def test_matching_mean_offered_demand_and_conservation(self):
        a, b = self.dispatch(False), self.dispatch(True)
        self.assertAlmostEqual(a.offered_rps.mean(), b.offered_rps.mean())
        np.testing.assert_allclose(b.node_offered_arrival_rps, [10, 20])
        np.testing.assert_allclose(b.offered_rps, b.admitted_rps + b.excess_rps)

    def test_guard_records_excess_but_does_not_call_it_completed(self):
        source = self.traffic.copy()
        source["count"] = [0, 100]
        service = (FixedService("s", "d", 4, 20, 25),)
        out = make_dispatch(source, service, {"s": "count"}, 1.0, variable=True)
        self.assertGreater(out.excess_rps.sum(), 0)
        self.assertFalse(out.completed_throughput_modeled.any())
        with self.assertRaisesRegex(ValueError, "Unsupported/zero-load"):
            validate_dispatch(self.generator, out, 0.1)

    def test_invalid_timeline_and_counts_are_rejected(self):
        for column, values in (("time_s", [0, 2]), ("count", [1, -1]), ("bin_s", [1, 0])):
            traffic = self.traffic.copy()
            traffic[column] = values
            with self.assertRaises(ValueError):
                self.dispatch(traffic=traffic)

    def test_state_survives_block_boundary_without_demeaning(self):
        coeff = R3Coefficients(0.8, 0.0, 0.0, np.array([0., 1., 0.]), np.zeros(3), np.zeros(3))
        first, state = advance_residual(coeff, np.array([1.0, -2.0]), 3, np.random.default_rng(1))
        second, end = advance_residual(coeff, state, 2, np.random.default_rng(2))
        np.testing.assert_allclose(first[0], [0.8, 0.64, 0.512])
        np.testing.assert_allclose(second[0], [0.4096, 0.32768])
        self.assertNotEqual(float(first[0].mean()), 0.0)
        np.testing.assert_allclose(end, second[:, -1])

    def test_coefficients_have_zero_mean_innovation(self):
        c = coefficients_at(self.generator, "d", 15)
        self.assertAlmostEqual(float(np.dot(c.weights, c.means_kw)), 0.0)
        self.assertGreater(c.innovation_variance, 0.0)
        with self.assertRaises(ValueError):
            coefficients_at(self.generator, "d", 0)

    def test_reference_design_is_input_scale_not_output_rescale(self):
        target = 4 * float(self.generator.m3_parameters["d"].predict(15)) / 1000
        scale = reference_design_scale(self.generator, self.services, target)
        self.assertAlmostEqual(scale, 1.0, places=7)

    def test_constant_trace_produces_identical_matched_cases(self):
        t = self.traffic.copy()
        t["count"] = [15, 15]
        b = make_dispatch(t, self.services, {"s": "count"}, 1, variable=False)
        c = make_dispatch(t, self.services, {"s": "count"}, 1, variable=True)
        pb, _, _ = simulate_case(self.generator, b, self.services, case="B", scenario_count=2, burn_in_s=1)
        pc, _, _ = simulate_case(self.generator, c, self.services, case="C", scenario_count=2, burn_in_s=1)
        np.testing.assert_array_equal(pb.total_it_mw, pc.total_it_mw)

    def test_small_simulation_is_reproducible_positive_and_additive(self):
        d = self.dispatch()
        first, nodes, audit = simulate_case(self.generator, d, self.services, case="C", scenario_count=2, burn_in_s=1)
        second, _, _ = simulate_case(self.generator, d, self.services, case="C", scenario_count=2, burn_in_s=1)
        self.assertEqual(len(first), 40)
        np.testing.assert_array_equal(first.total_it_mw, second.total_it_mw)
        np.testing.assert_allclose(first.total_it_mw, first.m3_total_mw + first.residual_total_mw)
        self.assertTrue((audit.minimum_node_reported_power_kw >= 0).all())
        self.assertEqual(len(nodes), 20)
        self.assertEqual(first.traffic_boundary.sum(), 2)

    def test_boundary_ramp_is_excluded_from_within_bin_score(self):
        p = np.r_[np.ones(20), np.full(20, 2.)]
        boundary = np.zeros(40, bool)
        boundary[20] = True
        frame = pd.DataFrame({"total_it_mw": p, "traffic_boundary": boundary})
        scores = profile_metrics(frame, 0.1, 2)
        self.assertAlmostEqual(scores["max_abs_ramp_0p1s_mw_per_s"], 10)
        self.assertAlmostEqual(scores["max_within_bin_abs_ramp_0p1s_mw_per_s"], 0)
        self.assertAlmostEqual(scores["max_within_bin_abs_ramp_1p0s_mw_per_s"], 0)

    def test_sampling_interval_cannot_be_changed_silently(self):
        with self.assertRaises(ValueError):
            validate_dispatch(self.generator, self.dispatch(), 1.0)

    def test_noninteger_node_type_and_conflicting_dispatch_rejected(self):
        with self.assertRaises(ValueError):
            FixedService("s", "d", 4.0, 15).validate()
        d = self.dispatch()
        d["node_count"] = 999
        with self.assertRaisesRegex(ValueError, "does not match"):
            simulate_case(self.generator, d, self.services, case="bad", burn_in_s=0)


if __name__ == "__main__":
    unittest.main()
