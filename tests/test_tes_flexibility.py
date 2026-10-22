"""Independent numerical checks for the conditional TES flexibility study."""
import copy
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aidc_facility.tes_flexibility import (
    duration_limits,
    evaluate_cycle,
    marginal_pcc_coefficients,
    pcc_from_thermal,
)


class TestTESFlexibility(unittest.TestCase):
    def setUp(self):
        self.dt = 1 / 12
        self.tes = {
            "energy_capacity_mwh_th": 10.0,
            "min_soc": 0.1,
            "max_soc": 1.0,
            "charge_limit_mw_th": 3.0,
            "discharge_limit_mw_th": 3.0,
            "charge_efficiency": 0.9,
            "discharge_efficiency": 0.9,
            "hourly_retention_fraction": 1.0,
            "capacity_loss_fraction_per_hour": 0.0004,
            "pump_mw_per_mw_th": 0.01,
        }
        self.legacy = {
            "electrical": {"ups_efficiency": 0.97,
                           "transformer_efficiency": 0.99,
                           "auxiliary_mw": 0.1},
            "plant": {"chiller_capacity_mw_th": 26.0,
                      "condenser_pump_mw_per_mw_heat": 0.005,
                      "tower_fan_mw_per_mw_heat": 0.015},
        }

    def trace(self, n=240, thermal=18.0):
        return [{"chiller_mw_th": thermal, "fan_mw": 0.5,
                 "water_pump_mw": 0.15, "it_mw": 17.0}
                for _ in range(n)]

    def cycle(self, rows=None, **kwargs):
        args = {"base_rows": self.trace() if rows is None else rows,
                "start_index": 0, "initial_soc": 0.5,
                "discharge_rate": 3.0, "tes_config": self.tes,
                "legacy": self.legacy, "dt_h": self.dt, "cop": 5.0}
        args.update(kwargs)
        return evaluate_cycle(**args)

    def test_half_and_full_soc_duration_and_five_minute_floor(self):
        for energy, steps in ((5.0, 14), (10.0, 32)):
            with self.subTest(energy=energy):
                result = duration_limits(energy, 3.0, self.dt, self.tes)
                expected = (energy - 1.0) / (3.0 / 0.9 + 0.004)
                self.assertAlmostEqual(result["continuous_duration_h"], expected, places=12)
                self.assertEqual(result["full_steps"], steps)
                self.assertAlmostEqual(result["full_interval_duration_h"], steps / 12)
                self.assertGreaterEqual(result["unused_fractional_duration_h"], 0)
                self.assertLess(result["unused_fractional_duration_h"], self.dt)
                self.assertLessEqual(steps * self.dt, expected)
                self.assertGreater((steps + 1) * self.dt, expected)

    def test_exact_interval_duration_is_not_rounded_down(self):
        c = dict(self.tes, capacity_loss_fraction_per_hour=0.0)
        # 4 MWhth available, 0.9 discharge efficiency, 3 MWth output: 1.2 h.
        result = duration_limits(5.0, 3.0, 0.1, c)
        self.assertEqual(result["full_steps"], 12)
        self.assertAlmostEqual(result["unused_fractional_duration_h"], 0.0)

    def test_minimum_soc_has_no_full_dispatch_interval(self):
        result = duration_limits(1.0, 3.0, self.dt, self.tes)
        self.assertEqual(result["continuous_duration_h"], 0)
        self.assertEqual(result["full_steps"], 0)
        cycle = self.cycle(initial_soc=0.1)
        self.assertEqual(cycle["status"], "no_full_discharge_interval")
        self.assertFalse(cycle["completed"])
        self.assertEqual(cycle["rows"], [])

    def test_zero_negative_and_over_rating_discharge_are_rejected(self):
        for rate in (0.0, -1.0, 3.001, math.nan, math.inf):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                duration_limits(5.0, rate, self.dt, self.tes)

    def test_duration_rejects_invalid_state_interval_and_nonconstant_loss(self):
        for energy in (0.99, 10.01, math.nan):
            with self.subTest(energy=energy), self.assertRaises(ValueError):
                duration_limits(energy, 3.0, self.dt, self.tes)
        for dt in (0, -1, math.inf):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                duration_limits(5.0, 3.0, dt, self.tes)
        with self.assertRaises(ValueError):
            duration_limits(5.0, 3.0, self.dt,
                            dict(self.tes, hourly_retention_fraction=0.99))

    def test_marginal_coefficients_match_independent_equipment_accounting(self):
        for cop in (4.0, 5.0, 6.0):
            for k in (0.0, 0.01, 0.02):
                with self.subTest(cop=cop, pump=k):
                    a, b, eta = 0.005, 0.015, 0.99
                    g = 1 / cop + (a + b * (1 + a)) * (1 + 1 / cop)
                    expected_down = ((1 - k) * g - k) / eta
                    expected_up = ((1 + k) * g + k) / eta
                    coeff = marginal_pcc_coefficients(self.legacy, cop, k)
                    self.assertAlmostEqual(coeff["reduction_mwe_per_mwth"], expected_down, places=12)
                    self.assertAlmostEqual(coeff["increase_mwe_per_mwth"], expected_up, places=12)
                    for base in (self.trace(1, 12)[0], self.trace(1, 21)[0]):
                        idle = pcc_from_thermal(base, 0, 0, self.legacy, cop, k)["pcc_mw"]
                        discharge = pcc_from_thermal(base, 0, 3, self.legacy, cop, k)["pcc_mw"]
                        charge = pcc_from_thermal(base, 3, 0, self.legacy, cop, k)["pcc_mw"]
                        self.assertAlmostEqual(idle - discharge, 3 * expected_down, places=11)
                        self.assertAlmostEqual(charge - idle, 3 * expected_up, places=11)

    def test_accounting_rejects_invalid_cop_or_simultaneous_ports(self):
        base = self.trace(1)[0]
        for cop in (0, -1, math.nan):
            with self.subTest(cop=cop), self.assertRaises(ValueError):
                pcc_from_thermal(base, 0, 0, self.legacy, cop, 0.01)
        for charge, discharge in ((1, 1), (-1, 0), (0, -1)):
            with self.subTest(charge=charge, discharge=discharge), self.assertRaises(ValueError):
                pcc_from_thermal(base, charge, discharge, self.legacy, 5, 0.01)

    def test_complete_cycle_has_equal_endpoints_and_paid_baseline_recovery(self):
        result = self.cycle()
        self.assertTrue(result["completed"])
        self.assertEqual(result["status"], "complete")
        self.assertTrue(all(result["checks"].values()))
        self.assertAlmostEqual(result["full_discharge_duration_h"], 14 / 12)
        rows = result["rows"]
        for prefix in ("", "baseline_"):
            self.assertAlmostEqual(rows[0][prefix + "energy_start_mwh_th"], 5.0)
            self.assertAlmostEqual(rows[-1][prefix + "energy_end_mwh_th"], 5.0)
            charge = sum(r[prefix + "charge_mw_th"] * self.dt for r in rows)
            discharge = sum(r[prefix + "discharge_mw_th"] * self.dt for r in rows)
            expected_change = 0.9 * charge - discharge / 0.9 - 0.004 * len(rows) * self.dt
            self.assertAlmostEqual(expected_change, 0.0, places=11)
        self.assertGreater(rows[-1]["baseline_charge_mw_th"], 0)
        self.assertGreater(result["net_electricity_increase_mwh"], 0)
        self.assertAlmostEqual(result["net_electricity_increase_mwh"],
                               result["recovery_electricity_increase_mwh"]
                               - result["discharge_electricity_reduction_mwh"], places=12)

    def test_recovery_uses_fractional_final_power_not_state_reset(self):
        result = self.cycle()
        recovery = [r for r in result["rows"] if r["phase"] == "recovery"]
        self.assertGreater(len(recovery), 1)
        self.assertTrue(all(abs(r["charge_mw_th"] - 3.0) < 1e-10 for r in recovery[:-1]))
        final = recovery[-1]
        self.assertGreater(final["charge_mw_th"], 0)
        self.assertLess(final["charge_mw_th"], 3.0)
        expected = (5.0 - final["energy_start_mwh_th"] + 0.004 * self.dt) / (0.9 * self.dt)
        self.assertAlmostEqual(final["charge_mw_th"], expected, places=11)
        self.assertAlmostEqual(final["energy_end_mwh_th"], 5.0, places=11)

    def test_forced_overlong_discharge_reports_shortfall_and_maintains_reserve(self):
        result = self.cycle(discharge_steps=40)
        self.assertTrue(result["completed"])
        self.assertGreater(result["discharge_shortfall_mwh_th"], 0)
        self.assertFalse(result["checks"]["full_discharge_delivered"])
        self.assertTrue(result["checks"]["soc_and_ratings"])
        self.assertTrue(result["checks"]["same_initial_final_energy"])
        maintenance = [r for r in result["rows"] if r["reserve_maintenance"]]
        self.assertTrue(maintenance)
        self.assertTrue(all(r["charge_mw_th"] > 0 and r["discharge_mw_th"] == 0 for r in maintenance))

    def test_low_cooling_demand_limits_discharge_and_reports_shortfall(self):
        result = self.cycle(rows=self.trace(240, 0.2), discharge_steps=4)
        self.assertTrue(result["completed"])
        self.assertFalse(result["checks"]["full_discharge_delivered"])
        self.assertGreater(result["discharge_shortfall_mwh_th"], 0)
        for row in result["rows"]:
            if row["phase"] == "discharge":
                self.assertAlmostEqual(row["discharge_mw_th"], 0.2 / 0.99)
                self.assertGreaterEqual(row["chiller_mw_th"], -1e-10)

    def test_charge_headroom_limits_recovery_without_exceeding_capacity(self):
        rows = self.trace()
        for row in rows[4:]:
            row["chiller_mw_th"] = 25.0
        result = self.cycle(rows=rows, discharge_steps=4)
        normal = self.cycle(discharge_steps=4)
        self.assertTrue(result["completed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertGreater(result["recovery_duration_h"], normal["recovery_duration_h"])
        maximum = (26 - 25) / 1.01
        for row in result["rows"]:
            if row["phase"] == "recovery":
                self.assertLessEqual(row["charge_mw_th"], maximum + 1e-10)
                self.assertLessEqual(row["chiller_mw_th"], 26 + 1e-10)

    def test_no_charge_headroom_cannot_claim_recovery(self):
        rows = self.trace(30, 26.0)
        rows[0]["chiller_mw_th"] = 18.0
        result = self.cycle(rows=rows, discharge_steps=1)
        self.assertFalse(result["completed"])
        self.assertEqual(result["status"], "trace_exhausted")
        self.assertLess(result["rows"][-1]["energy_end_mwh_th"], 5.0)
        self.assertTrue(all(r["charge_mw_th"] == 0 for r in result["rows"] if r["phase"] == "recovery"))

    def test_trace_exhaustion_preserves_source_indices_without_wrapping(self):
        result = self.cycle(rows=self.trace(10), start_index=8)
        self.assertFalse(result["completed"])
        self.assertEqual(result["status"], "trace_exhausted")
        self.assertEqual([r["source_index"] for r in result["rows"]], [8, 9])
        self.assertEqual(len(result["rows"]), 2)
        self.assertNotIn("net_electricity_increase_mwh", result)

    def test_input_data_and_configs_are_not_mutated(self):
        rows = self.trace()
        originals = copy.deepcopy((rows, self.tes, self.legacy))
        self.cycle(rows=rows)
        self.assertEqual((rows, self.tes, self.legacy), originals)

    def test_invalid_trace_index_or_discharge_step_count_is_rejected(self):
        for index in (-1, 240, 1.2):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.cycle(start_index=index)
        for steps in (-1, 2.5):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                self.cycle(discharge_steps=steps)


if __name__ == "__main__":
    unittest.main()
