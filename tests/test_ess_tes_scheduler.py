"""Independent accounting and feasibility tests for joint ESS--TES dispatch.

These synthetic cases are deliberately small and analytically interpretable.
The replay below reconstructs PCC electricity and both inventories directly;
it does not rely on the new scheduler's own replay or verification flags.
"""

import copy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aidc_facility.ess_tes_scheduler import solve_joint_schedule
from aidc_facility.tes_scheduler import solve_schedule as solve_tes_schedule


class TestESSTESScheduler(unittest.TestCase):
    def setUp(self):
        self.dt = 1 / 12
        self.tes = {
            "energy_capacity_mwh_th": 10.0,
            "min_soc": 0.1, "max_soc": 1.0,
            "charge_limit_mw_th": 3.0, "discharge_limit_mw_th": 3.0,
            "charge_efficiency": 0.9, "discharge_efficiency": 0.9,
            "hourly_retention_fraction": 1.0,
            "capacity_loss_fraction_per_hour": 0.0004,
            "pump_mw_per_mw_th": 0.01,
        }
        self.ess = {
            "energy_capacity_mwh": 4.0,
            "min_soc": 0.1, "max_soc": 1.0,
            "charge_limit_mw": 1.0, "discharge_limit_mw": 1.0,
            "charge_efficiency": 0.95, "discharge_efficiency": 0.95,
            "hourly_retention_fraction": 1.0,
        }
        self.legacy = {
            "electrical": {"ups_efficiency": 0.97,
                           "transformer_efficiency": 0.99,
                           "auxiliary_mw": 0.1},
            "plant": {"chiller_capacity_mw_th": 26.0,
                      "condenser_pump_mw_per_mw_heat": 0.005,
                      "tower_fan_mw_per_mw_heat": 0.015},
        }

    def trace(self, n=12, thermal=18.0):
        return [{"chiller_mw_th": thermal, "fan_mw": 0.5,
                 "water_pump_mw": 0.15, "it_mw": 17.0}
                for _ in range(n)]

    def facility_pcc(self, base, qc=0.0, qd=0.0, tes=None, legacy=None, cop=5.0):
        """Direct original facility equation, without scheduler helper calls."""
        t = self.tes if tes is None else tes
        legacy = self.legacy if legacy is None else legacy
        electrical, plant = legacy["electrical"], legacy["plant"]
        tes_pump = t["pump_mw_per_mw_th"] * (qc + qd)
        heat = base["chiller_mw_th"] + tes_pump + qc - qd
        chiller = heat / cop
        condenser_pump = plant["condenser_pump_mw_per_mw_heat"] * (heat + chiller)
        tower = plant["tower_fan_mw_per_mw_heat"] * (heat + chiller + condenser_pump)
        cooling = base["fan_mw"] + base["water_pump_mw"] + tes_pump + chiller + condenser_pump + tower
        return (base["it_mw"] / electrical["ups_efficiency"]
                + electrical["auxiliary_mw"] + cooling) / electrical["transformer_efficiency"]

    def solve(self, rows=None, caps=None, **overrides):
        rows = self.trace() if rows is None else rows
        caps = [self.facility_pcc(r) + 3 for r in rows] if caps is None else caps
        options = {
            "base_rows": rows, "pcc_upper_mw": caps,
            "tes_config": self.tes, "ess_config": self.ess, "legacy": self.legacy,
            "dt_h": self.dt,
            "initial_tes_soc": 0.5, "terminal_tes_soc": 0.5,
            "initial_ess_soc": 0.5, "terminal_ess_soc": 0.5,
            "cop": 5.0, "use_tes": True, "use_ess": True,
            "objective": "energy", "time_limit_s": 30,
        }
        options.update(overrides)
        return solve_joint_schedule(**options)

    def assert_replay(self, result, rows, caps, *, tes=None, ess=None, dt=None,
                      use_tes=True, use_ess=True, initial_tes_soc=0.5,
                      terminal_tes_soc=0.5, initial_ess_soc=0.5,
                      terminal_ess_soc=0.5, legacy=None):
        tes, ess = self.tes if tes is None else tes, self.ess if ess is None else ess
        dt = self.dt if dt is None else dt
        legacy = self.legacy if legacy is None else legacy
        self.assertTrue(result["success"], result)
        self.assertTrue(result["feasible"], result)
        self.assertTrue(result["optimal"], result)
        self.assertTrue(result["checks"])
        self.assertTrue(all(result["checks"].values()), result["checks"])
        # All outputs, including failure metadata, must be portable finite JSON.
        json.dumps(result, allow_nan=False)
        schedule = result["schedule"]
        n = len(rows)
        self.assertEqual(len(result["rows"]), n)
        for key in ("tes_charge_mw_th", "tes_discharge_mw_th", "ess_charge_mw",
                    "ess_discharge_mw", "pcc_mw", "pcc_upper_mw", "no_storage_pcc_mw"):
            self.assertEqual(len(schedule[key]), n, key)
        for key in ("tes_energy_mwh_th", "ess_energy_mwh"):
            self.assertEqual(len(schedule[key]), n + 1, key)
        et = initial_tes_soc * tes["energy_capacity_mwh_th"] if use_tes else 0.0
        eb = initial_ess_soc * ess["energy_capacity_mwh"] if use_ess else 0.0
        self.assertAlmostEqual(schedule["tes_energy_mwh_th"][0], et, places=6)
        self.assertAlmostEqual(schedule["ess_energy_mwh"][0], eb, places=6)
        retention = ess.get("hourly_retention_fraction", 1.0)
        decay = -math.log(retention)
        alpha = retention ** dt
        beta = -math.expm1(-decay * dt) / decay if decay else dt
        for k, (base, cap) in enumerate(zip(rows, caps)):
            qc, qd = schedule["tes_charge_mw_th"][k], schedule["tes_discharge_mw_th"][k]
            pc, pd = schedule["ess_charge_mw"][k], schedule["ess_discharge_mw"][k]
            for a, b, ca, cb in ((qc, qd, tes["charge_limit_mw_th"], tes["discharge_limit_mw_th"]),
                                  (pc, pd, ess["charge_limit_mw"], ess["discharge_limit_mw"])):
                self.assertGreaterEqual(min(a, b), -1e-7)
                self.assertLessEqual(a, ca + 1e-7)
                self.assertLessEqual(b, cb + 1e-7)
                self.assertLessEqual(min(a, b), 1e-7)
            if use_tes:
                et += (tes["charge_efficiency"] * qc - qd / tes["discharge_efficiency"]
                       - tes["capacity_loss_fraction_per_hour"] * tes["energy_capacity_mwh_th"]) * dt
                self.assertGreaterEqual(et, tes["min_soc"] * tes["energy_capacity_mwh_th"] - 1e-7)
                self.assertLessEqual(et, tes["max_soc"] * tes["energy_capacity_mwh_th"] + 1e-7)
            else:
                self.assertAlmostEqual(qc + qd, 0.0, places=8)
            if use_ess:
                eb = alpha * eb + beta * (ess["charge_efficiency"] * pc - pd / ess["discharge_efficiency"])
                self.assertGreaterEqual(eb, ess["min_soc"] * ess["energy_capacity_mwh"] - 1e-7)
                self.assertLessEqual(eb, ess["max_soc"] * ess["energy_capacity_mwh"] + 1e-7)
            else:
                self.assertAlmostEqual(pc + pd, 0.0, places=8)
            self.assertAlmostEqual(schedule["tes_energy_mwh_th"][k + 1], et, places=6)
            self.assertAlmostEqual(schedule["ess_energy_mwh"][k + 1], eb, places=6)
            heat = base["chiller_mw_th"] + tes["pump_mw_per_mw_th"] * (qc + qd) + qc - qd
            self.assertGreaterEqual(heat, -1e-7)
            self.assertLessEqual(heat, legacy["plant"]["chiller_capacity_mw_th"] + 1e-7)
            # ESS commands already refer to AC at PCC: no second efficiency factor.
            expected_pcc = self.facility_pcc(base, qc, qd, tes=tes, legacy=legacy) + pc - pd
            self.assertAlmostEqual(schedule["pcc_mw"][k], expected_pcc, places=6)
            self.assertAlmostEqual(schedule["no_storage_pcc_mw"][k], self.facility_pcc(base, legacy=legacy), places=6)
            self.assertAlmostEqual(schedule["pcc_upper_mw"][k], cap, places=7)
            self.assertGreaterEqual(expected_pcc, -1e-7)
            self.assertLessEqual(expected_pcc, cap + 1e-7)
        self.assertAlmostEqual(et, terminal_tes_soc * tes["energy_capacity_mwh_th"] if use_tes else 0, places=6)
        self.assertAlmostEqual(eb, terminal_ess_soc * ess["energy_capacity_mwh"] if use_ess else 0, places=6)

    def assert_infeasible(self, result):
        self.assertEqual(result["status"], "infeasible", result)
        self.assertFalse(result["success"])
        self.assertFalse(result["feasible"])
        self.assertFalse(result["optimal"])
        self.assertIsNone(result["schedule"])
        self.assertEqual(result["rows"], [])
        json.dumps(result, allow_nan=False)

    def test_no_storage_is_exact_baseline_with_no_inventory_or_standby(self):
        rows = self.trace(4)
        caps = [self.facility_pcc(r) for r in rows]
        result = self.solve(rows, caps, use_tes=False, use_ess=False)
        self.assert_replay(result, rows, caps, use_tes=False, use_ess=False)

    def test_no_storage_cannot_silently_curtail_to_meet_cap(self):
        rows = self.trace(2)
        caps = [self.facility_pcc(r) - 0.01 for r in rows]
        self.assert_infeasible(self.solve(rows, caps, use_tes=False, use_ess=False))

    def test_ess_only_no_need_to_charge_for_absent_tes_losses(self):
        rows = self.trace(12)
        caps = [self.facility_pcc(r) for r in rows]
        result = self.solve(rows, caps, use_tes=False)
        self.assert_replay(result, rows, caps, use_tes=False)
        self.assertAlmostEqual(sum(result["schedule"]["ess_charge_mw"]), 0.0, places=8)
        self.assertAlmostEqual(sum(result["schedule"]["ess_discharge_mw"]), 0.0, places=8)

    def test_ess_ac_round_trip_and_no_extra_transformer_or_ups_factor(self):
        rows = self.trace(3)
        caps = [self.facility_pcc(rows[0]) - 0.5] + [self.facility_pcc(r) + 2 for r in rows[1:]]
        result = self.solve(rows, caps, use_tes=False, dt_h=1.0)
        self.assert_replay(result, rows, caps, use_tes=False, dt=1.0)
        s = result["schedule"]
        self.assertAlmostEqual(s["ess_discharge_mw"][0], 0.5, places=7)
        self.assertAlmostEqual(sum(s["ess_discharge_mw"]), 0.5, places=7)
        self.assertAlmostEqual(sum(s["ess_charge_mw"]), 0.5 / (0.95 * 0.95), places=7)
        self.assertAlmostEqual(s["pcc_mw"][0], self.facility_pcc(rows[0]) - 0.5, places=7)

    def test_ess_pcc_reference_survives_changed_facility_efficiency(self):
        legacy = copy.deepcopy(self.legacy)
        legacy["electrical"].update(ups_efficiency=0.8, transformer_efficiency=0.8)
        rows = self.trace(3)
        caps = [self.facility_pcc(r, legacy=legacy) + 2 for r in rows]
        caps[0] = self.facility_pcc(rows[0], legacy=legacy) - 0.75
        result = self.solve(rows, caps, legacy=legacy, use_tes=False, dt_h=1.0)
        self.assert_replay(result, rows, caps, legacy=legacy, use_tes=False, dt=1.0)
        self.assertAlmostEqual(result["schedule"]["ess_discharge_mw"][0], 0.75, places=7)

    def test_ess_terminal_target_is_paid_in_actual_pcc_charging(self):
        rows = self.trace(1)
        caps = [self.facility_pcc(rows[0]) + 1]
        result = self.solve(rows, caps, use_tes=False, dt_h=1.0, terminal_ess_soc=0.6)
        self.assert_replay(result, rows, caps, use_tes=False, dt=1.0, terminal_ess_soc=0.6)
        self.assertAlmostEqual(result["schedule"]["ess_charge_mw"][0], 0.4 / 0.95, places=7)

    def test_ess_hourly_retention_uses_exact_constant_input_solution(self):
        rows = self.trace(1)
        caps = [self.facility_pcc(rows[0]) + 1]
        ess = dict(self.ess, hourly_retention_fraction=0.98)
        result = self.solve(rows, caps, ess_config=ess, use_tes=False, dt_h=0.5)
        self.assert_replay(result, rows, caps, ess=ess, use_tes=False, dt=0.5)
        beta = (1 - 0.98 ** 0.5) / -math.log(0.98)
        required = 2 * (1 - 0.98 ** 0.5) / (0.95 * beta)
        self.assertAlmostEqual(result["schedule"]["ess_charge_mw"][0], required, places=7)

    def test_tes_only_reproduces_previous_solver_energy(self):
        rows = self.trace(36)
        caps = [self.facility_pcc(r) + 2 for r in rows]
        for k in range(12, 24):
            caps[k] = self.facility_pcc(rows[k], qd=2.0)
        old = solve_tes_schedule(rows, caps, self.tes, self.legacy, dt_h=self.dt,
                                 initial_soc=0.5, terminal_soc=0.5, time_limit_s=30)
        result = self.solve(rows, caps, use_ess=False)
        self.assertTrue(old["feasible"], old)
        self.assert_replay(result, rows, caps, use_ess=False)
        self.assertAlmostEqual(sum(result["schedule"]["pcc_mw"]), sum(old["schedule"]["pcc_mw"]), places=5)
        self.assertAlmostEqual(sum(result["schedule"]["tes_discharge_mw_th"]),
                               sum(old["schedule"]["discharge_mw_th"]), places=5)

    def test_joint_can_meet_a_cap_that_neither_storage_alone_can(self):
        rows = self.trace(5)
        caps = [self.facility_pcc(r) + 3 for r in rows]
        caps[2] = self.facility_pcc(rows[2]) - 1.4
        for use_tes, use_ess in ((False, True), (True, False)):
            self.assert_infeasible(self.solve(rows, caps, use_tes=use_tes, use_ess=use_ess, dt_h=1.0))
        result = self.solve(rows, caps, dt_h=1.0)
        self.assert_replay(result, rows, caps, dt=1.0)
        self.assertGreater(result["schedule"]["ess_discharge_mw"][2], 0.0)
        self.assertGreater(result["schedule"]["tes_discharge_mw_th"][2], 0.0)

    def test_joint_power_beyond_combined_ratings_is_infeasible(self):
        rows = self.trace(3)
        caps = [self.facility_pcc(r) + 3 for r in rows]
        caps[1] = self.facility_pcc(rows[1], qd=3.0) - 1.0 - 0.01
        self.assert_infeasible(self.solve(rows, caps, dt_h=1.0))

    def test_prolonged_deficit_is_energy_infeasible_despite_power_headroom(self):
        rows = self.trace(10)
        caps = [self.facility_pcc(r) - 1.4 for r in rows]
        # Instantaneous available reduction >1.4 MW, but inventories cannot
        # sustain a 14 MWh_e import deficit. Initial/final SOC are explicit.
        self.assertGreater(self.facility_pcc(rows[0]) - self.facility_pcc(rows[0], qd=3) + 1, 1.4)
        self.assert_infeasible(self.solve(rows, caps, dt_h=1.0,
                                         initial_tes_soc=1.0, terminal_tes_soc=0.1,
                                         initial_ess_soc=1.0, terminal_ess_soc=0.1))

    def test_terminal_ess_recharge_cannot_ignore_pcc_cap(self):
        rows = self.trace(2)
        p0 = self.facility_pcc(rows[0])
        self.assert_infeasible(self.solve(rows, [p0 - 0.5, p0], use_tes=False, dt_h=1.0))
        caps = [p0 - 0.5, p0 + 0.5 / (0.95 * 0.95)]
        result = self.solve(rows, caps, use_tes=False, dt_h=1.0)
        self.assert_replay(result, rows, caps, use_tes=False, dt=1.0)

    def test_tes_chiller_capacity_is_enforced_in_joint_model(self):
        rows = self.trace(2, thermal=26.0)
        rows[0]["chiller_mw_th"] = 18.0
        caps = [self.facility_pcc(rows[0], qd=3.0), self.facility_pcc(rows[1]) + 3]
        self.assert_infeasible(self.solve(rows, caps, use_ess=False, dt_h=1.0))

    def test_surplus_ess_discharge_cannot_be_exported_implicitly(self):
        rows = [{"it_mw": 0.1, "fan_mw": 0.0, "water_pump_mw": 0.0,
                 "chiller_mw_th": 0.0}]
        # The terminal target would require 0.475 MW of discharge, greater
        # than this tiny site's demand. The import-only model must reject it.
        self.assertLess(self.facility_pcc(rows[0]), 0.475)
        result = self.solve(rows, [1.0], use_tes=False, dt_h=1.0,
                            initial_ess_soc=0.5, terminal_ess_soc=0.375)
        self.assert_infeasible(result)

    def test_scalar_cap_and_default_retention_are_supported(self):
        rows = self.trace(6)
        ess = dict(self.ess)
        del ess["hourly_retention_fraction"]
        result = self.solve(rows, 30.0, ess_config=ess, use_tes=False)
        self.assert_replay(result, rows, [30.0] * len(rows), ess=ess, use_tes=False)

    def test_zero_power_ports_do_not_allow_free_soc_changes(self):
        ess = dict(self.ess, charge_limit_mw=0.0, discharge_limit_mw=0.0)
        result = self.solve(self.trace(1), [30.0], ess_config=ess, use_tes=False,
                            initial_ess_soc=0.5, terminal_ess_soc=0.6, dt_h=1.0)
        self.assert_infeasible(result)

    def test_inputs_are_not_mutated(self):
        rows = self.trace(6)
        caps = [self.facility_pcc(r) + 2 for r in rows]
        before = copy.deepcopy((rows, caps, self.tes, self.ess, self.legacy))
        result = self.solve(rows, caps)
        self.assert_replay(result, rows, caps)
        self.assertEqual((rows, caps, self.tes, self.ess, self.legacy), before)

    def test_invalid_ess_parameters_are_rejected(self):
        for key in ("energy_capacity_mwh", "charge_limit_mw", "discharge_limit_mw",
                    "charge_efficiency", "discharge_efficiency", "hourly_retention_fraction",
                    "min_soc", "max_soc"):
            for value in (math.nan, math.inf, -math.inf, True, -1.0):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.solve(self.trace(1), [30.0], ess_config=dict(self.ess, **{key: value}))
        for key in ("energy_capacity_mwh", "charge_efficiency", "discharge_efficiency", "hourly_retention_fraction"):
            with self.subTest(key=key, value=0), self.assertRaises(ValueError):
                self.solve(self.trace(1), [30.0], ess_config=dict(self.ess, **{key: 0.0}))
        for key in ("charge_efficiency", "discharge_efficiency", "hourly_retention_fraction", "max_soc"):
            with self.subTest(key=key, value=1.1), self.assertRaises(ValueError):
                self.solve(self.trace(1), [30.0], ess_config=dict(self.ess, **{key: 1.1}))
        with self.assertRaises(ValueError):
            self.solve(self.trace(1), [30.0], ess_config=dict(self.ess, min_soc=0.8, max_soc=0.7))

    def test_invalid_times_endpoints_or_objective_are_rejected(self):
        for key in ("dt_h", "cop", "time_limit_s"):
            for value in (0.0, -1.0, math.nan, math.inf, True):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.solve(self.trace(1), [30.0], **{key: value})
        for key in ("initial_tes_soc", "terminal_tes_soc", "initial_ess_soc", "terminal_ess_soc"):
            for value in (0.09, 1.01, math.nan, math.inf, True):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.solve(self.trace(1), [30.0], **{key: value})
        with self.assertRaises(ValueError):
            self.solve(self.trace(1), [30.0], objective="tariff")
        for key in ("use_tes", "use_ess"):
            for value in (0, 1, "False", None):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.solve(self.trace(1), [30.0], **{key: value})

    def test_nonfinite_or_negative_profiles_and_bounds_are_rejected(self):
        for key in ("it_mw", "chiller_mw_th", "fan_mw", "water_pump_mw"):
            for value in (-1.0, math.nan, math.inf):
                rows = self.trace(1)
                rows[0][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.solve(rows, [30.0])
        for cap in (-1.0, math.nan, math.inf, True):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                self.solve(self.trace(1), [cap])
        for rows, caps in (([], []), (self.trace(2), [30.0]), (self.trace(1), [30.0, 30.0])):
            with self.subTest(rows=len(rows), caps=len(caps)), self.assertRaises(ValueError):
                self.solve(rows, caps)


if __name__ == "__main__":
    unittest.main()
