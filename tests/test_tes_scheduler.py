"""Independent cases for PCC-constrained, endpoint-balanced TES scheduling.

These tests use synthetic load/cap cases with analytically known feasibility;
they do not treat the optimiser's own checks as sufficient verification.
"""

import copy
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aidc_facility.tes_flexibility import pcc_from_thermal
from aidc_facility.tes_scheduler import solve_schedule


class TestTESScheduler(unittest.TestCase):
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
            "electrical": {
                "ups_efficiency": 0.97,
                "transformer_efficiency": 0.99,
                "auxiliary_mw": 0.1,
            },
            "plant": {
                "chiller_capacity_mw_th": 26.0,
                "condenser_pump_mw_per_mw_heat": 0.005,
                "tower_fan_mw_per_mw_heat": 0.015,
            },
        }

    def trace(self, n=24, thermal=18.0):
        return [{"chiller_mw_th": thermal, "fan_mw": 0.5,
                 "water_pump_mw": 0.15, "it_mw": 17.0}
                for _ in range(n)]

    def pcc(self, row, charge=0.0, discharge=0.0):
        return pcc_from_thermal(row, charge, discharge, self.legacy,
                                5.0, self.tes["pump_mw_per_mw_th"])["pcc_mw"]

    def solve(self, rows=None, caps=None, **kwargs):
        rows = self.trace() if rows is None else rows
        caps = [self.pcc(row) + 2 for row in rows] if caps is None else caps
        arguments = {
            "base_rows": rows,
            "pcc_upper_mw": caps,
            "tes_config": self.tes,
            "legacy": self.legacy,
            "dt_h": self.dt,
            "initial_soc": 0.5,
            "terminal_soc": 0.5,
            "cop": 5.0,
            "objective": "energy",
            "time_limit_s": 30,
        }
        arguments.update(kwargs)
        return solve_schedule(**arguments)

    def assert_replay(self, result, source, caps, initial_soc=0.5,
                      terminal_soc=0.5, tes=None, dt=None):
        c = self.tes if tes is None else tes
        dt = self.dt if dt is None else dt
        self.assertTrue(result["success"], result)
        self.assertTrue(result["feasible"], result)
        self.assertTrue(result["optimal"], result)
        self.assertEqual(len(result["rows"]), len(source))
        self.assertTrue(result["checks"], result)
        self.assertTrue(all(result["checks"].values()), result["checks"])
        expected_energy = initial_soc * c["energy_capacity_mwh_th"]
        for base, row, cap in zip(source, result["rows"], caps):
            charge, discharge = row["charge_mw_th"], row["discharge_mw_th"]
            self.assertGreaterEqual(charge, -1e-7)
            self.assertGreaterEqual(discharge, -1e-7)
            self.assertLessEqual(charge, c["charge_limit_mw_th"] + 1e-7)
            self.assertLessEqual(discharge, c["discharge_limit_mw_th"] + 1e-7)
            self.assertLessEqual(min(charge, discharge), 1e-7)
            self.assertAlmostEqual(row["energy_start_mwh_th"], expected_energy, places=6)
            expected_energy += (
                c["charge_efficiency"] * charge
                - discharge / c["discharge_efficiency"]
                - c["capacity_loss_fraction_per_hour"] * c["energy_capacity_mwh_th"]
            ) * dt
            self.assertAlmostEqual(row["energy_end_mwh_th"], expected_energy, places=6)
            self.assertGreaterEqual(expected_energy,
                                    c["min_soc"] * c["energy_capacity_mwh_th"] - 1e-7)
            self.assertLessEqual(expected_energy,
                                 c["max_soc"] * c["energy_capacity_mwh_th"] + 1e-7)
            pump = c["pump_mw_per_mw_th"] * (charge + discharge)
            heat = base["chiller_mw_th"] + pump + charge - discharge
            self.assertGreaterEqual(heat, -1e-7)
            self.assertLessEqual(heat, 26.0 + 1e-7)
            # Re-evaluate PCC independently of optimiser variables and metrics.
            electric = pcc_from_thermal(base, charge, discharge, self.legacy,
                                       5.0, c["pump_mw_per_mw_th"])["pcc_mw"]
            self.assertAlmostEqual(row["pcc_mw"], electric, places=6)
            self.assertLessEqual(electric, cap + 1e-7)
        self.assertAlmostEqual(expected_energy,
                               terminal_soc * c["energy_capacity_mwh_th"], places=6)

    def assert_infeasible(self, result):
        self.assertFalse(result["success"], result)
        self.assertFalse(result["feasible"], result)
        self.assertFalse(result["optimal"], result)

    def test_loose_cap_only_pays_standby_recovery(self):
        rows = self.trace(24)
        caps = [self.pcc(row) + 2 for row in rows]
        result = self.solve(rows, caps)
        self.assert_replay(result, rows, caps)
        charge = sum(r["charge_mw_th"] for r in result["rows"]) * self.dt
        discharge = sum(r["discharge_mw_th"] for r in result["rows"]) * self.dt
        self.assertAlmostEqual(discharge, 0.0, places=7)
        self.assertAlmostEqual(charge, 0.004 * 2 / 0.9, places=7)

    def test_lossless_loose_cap_does_not_cycle_storage(self):
        rows = self.trace(12)
        caps = [self.pcc(row) + 2 for row in rows]
        tes = dict(self.tes, capacity_loss_fraction_per_hour=0.0)
        result = self.solve(rows, caps, tes_config=tes)
        self.assert_replay(result, rows, caps, tes=tes)
        self.assertAlmostEqual(sum(r["charge_mw_th"] + r["discharge_mw_th"]
                                   for r in result["rows"]), 0.0, places=7)

    def test_peak_requires_precharge_and_paid_endpoint_restoration(self):
        rows = self.trace(36)
        caps = [self.pcc(row) + 2 for row in rows]
        for i in range(12, 24):
            caps[i] = self.pcc(rows[i], discharge=3.0)
        result = self.solve(rows, caps, initial_soc=0.2, terminal_soc=0.2)
        self.assert_replay(result, rows, caps, initial_soc=0.2, terminal_soc=0.2)
        dispatch = result["rows"]
        self.assertGreater(sum(r["charge_mw_th"] for r in dispatch[:12]), 1.0)
        for row in dispatch[12:24]:
            self.assertAlmostEqual(row["discharge_mw_th"], 3.0, places=6)
        self.assertGreater(sum(r["charge_mw_th"] for r in dispatch[24:]), 0.0)
        charged = sum(r["charge_mw_th"] for r in dispatch) * self.dt
        self.assertAlmostEqual(charged, (3.0 / 0.9 + 0.004 * 3) / 0.9, places=6)

    def test_instantaneous_reduction_beyond_rating_is_infeasible(self):
        rows = self.trace(24)
        caps = [self.pcc(row) + 2 for row in rows]
        caps[12] = self.pcc(rows[12], discharge=3) - 0.01
        self.assert_infeasible(self.solve(rows, caps))

    def test_sustained_request_can_be_power_feasible_but_energy_infeasible(self):
        rows = self.trace(40)
        caps = [self.pcc(row, discharge=3.0) for row in rows]
        # Each step is achievable at 3 MWth, but 40 steps require >9 MWhth.
        self.assertGreater(40 * self.dt * 3 / 0.9, 10 * (1.0 - 0.1))
        self.assert_infeasible(self.solve(rows, caps, initial_soc=1.0, terminal_soc=0.1))

    def test_no_recovery_chiller_headroom_makes_terminal_target_infeasible(self):
        rows = self.trace(2, thermal=26.0)
        rows[0]["chiller_mw_th"] = 18.0
        caps = [self.pcc(rows[0], discharge=3.0), self.pcc(rows[1]) + 2]
        self.assert_infeasible(self.solve(rows, caps, dt_h=1.0))

    def test_pcc_cap_blocks_recovery_even_with_chiller_headroom(self):
        rows = self.trace(2)
        caps = [self.pcc(rows[0], discharge=3.0), self.pcc(rows[1])]
        self.assert_infeasible(self.solve(rows, caps, dt_h=1.0))

    def test_terminal_soc_can_be_explicitly_different_from_initial(self):
        rows = self.trace(1)
        caps = [self.pcc(rows[0]) + 2]
        result = self.solve(rows, caps, initial_soc=0.5, terminal_soc=0.6, dt_h=1.0)
        self.assert_replay(result, rows, caps, initial_soc=0.5, terminal_soc=0.6, dt=1.0)
        self.assertAlmostEqual(result["rows"][0]["charge_mw_th"], (1 + 0.004) / 0.9,
                               places=7)

    def test_minimum_soc_requires_explicit_standby_compensation(self):
        rows = self.trace(3)
        caps = [self.pcc(row) + 2 for row in rows]
        result = self.solve(rows, caps, initial_soc=0.1, terminal_soc=0.1)
        self.assert_replay(result, rows, caps, initial_soc=0.1, terminal_soc=0.1)
        self.assertGreater(sum(r["charge_mw_th"] for r in result["rows"]), 0.0)

    def test_zero_cooling_demand_cannot_supply_discharge(self):
        rows = self.trace(2, thermal=0.0)
        caps = [self.pcc(rows[0]) - 0.01, self.pcc(rows[1]) + 2]
        self.assert_infeasible(self.solve(rows, caps))

    def test_inputs_are_not_mutated(self):
        rows = self.trace(12)
        caps = [self.pcc(row) + 2 for row in rows]
        original = copy.deepcopy((rows, caps, self.tes, self.legacy))
        self.solve(rows, caps)
        self.assertEqual((rows, caps, self.tes, self.legacy), original)

    def test_empty_or_mismatched_horizons_are_rejected(self):
        for rows, caps in (([], []), (self.trace(2), [30.0]),
                           (self.trace(1), [30.0, 30.0])):
            with self.subTest(rows=len(rows), caps=len(caps)), self.assertRaises(ValueError):
                self.solve(rows, caps)

    def test_nonfinite_or_negative_load_and_caps_are_rejected(self):
        for key in ("it_mw", "chiller_mw_th", "fan_mw", "water_pump_mw"):
            for value in (-1.0, math.nan, math.inf):
                rows = self.trace(1)
                rows[0][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.solve(rows, [30.0])
        for cap in (-1.0, math.nan, math.inf):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                self.solve(self.trace(1), [cap])

    def test_invalid_interval_cop_endpoints_or_objective_are_rejected(self):
        variants = []
        for name in ("dt_h", "cop", "time_limit_s"):
            variants.extend({name: value} for value in (0.0, -1.0, math.nan, math.inf))
        for name in ("initial_soc", "terminal_soc"):
            variants.extend({name: value} for value in (0.09, 1.01, math.nan, math.inf))
        variants.append({"objective": "unrecognized"})
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.solve(self.trace(1), [30.0], **variant)


if __name__ == "__main__":
    unittest.main()
