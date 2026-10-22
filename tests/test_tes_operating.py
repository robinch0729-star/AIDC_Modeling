import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aidc_facility.tes_operating import energy_step, target_charge_request, validate_tes


class TestTESOperating(unittest.TestCase):
    def setUp(self):
        self.c = {
            "energy_capacity_mwh_th": 20.0,
            "min_soc": 0.05,
            "max_soc": 0.95,
            "charge_limit_mw_th": 5.0,
            "discharge_limit_mw_th": 5.0,
            "charge_efficiency": 1.0,
            "discharge_efficiency": 1.0,
            "hourly_retention_fraction": 1.0,
            "pump_mw_per_mw_th": 0.02,
        }

    def step(self, energy=10.0, requested=0.0, dt=1.0, base=15.0, capacity=26.0, c=None):
        return energy_step(energy, requested, dt, base, capacity, self.c if c is None else c)

    def test_validation_returns_independent_copy(self):
        original = copy.deepcopy(self.c)
        result = validate_tes(self.c)
        result["max_soc"] = 0.8
        self.assertEqual(self.c, original)

    def test_known_lossless_charge_discharge(self):
        charge = self.step(requested=-3.0, dt=0.5)
        self.assertAlmostEqual(charge["energy_end_mwh_th"], 11.5)
        discharge = self.step(energy=11.5, requested=3.0, dt=0.5)
        self.assertAlmostEqual(discharge["energy_end_mwh_th"], 10.0)
        self.assertEqual(charge["standby_loss_mwh_th"], 0.0)

    def test_exact_hourly_retention_and_time_invariance(self):
        self.c["hourly_retention_fraction"] = 0.99
        one = self.step(energy=12.0, dt=2.0)
        energy = 12.0
        loss = 0.0
        for _ in range(24):
            out = self.step(energy=energy, dt=1 / 12)
            energy = out["energy_end_mwh_th"]
            loss += out["standby_loss_mwh_th"]
        self.assertAlmostEqual(energy, 12.0 * 0.99**2, places=12)
        self.assertAlmostEqual(energy, one["energy_end_mwh_th"], places=12)
        self.assertAlmostEqual(loss, one["standby_loss_mwh_th"], places=12)

    def test_capacity_fraction_hourly_loss_not_stored_energy_fraction(self):
        self.c.update(energy_capacity_mwh_th=10.0, capacity_loss_fraction_per_hour=0.0004)
        for start in (3.0, 7.0):
            out = self.step(energy=start)
            self.assertAlmostEqual(out["energy_end_mwh_th"], start - 0.004)
            self.assertAlmostEqual(out["capacity_loss_mwh_th"], 0.004)
            self.assertEqual(out["proportional_standby_loss_mwh_th"], 0)
            self.assertAlmostEqual(out["standby_loss_mwh_th"], 0.004)
            self.assertAlmostEqual(out["energy_balance_error_mwh_th"], 0)

    def test_combined_loss_time_invariance_and_restoration(self):
        self.c.update(hourly_retention_fraction=0.99, capacity_loss_fraction_per_hour=0.0004,
                      charge_efficiency=0.9)
        whole = self.step(energy=10.0, requested=-1.0, dt=2.0)
        energy, loss = 10.0, 0.0
        for _ in range(24):
            sub = self.step(energy=energy, requested=-1.0, dt=1 / 12)
            energy = sub["energy_end_mwh_th"]
            loss += sub["standby_loss_mwh_th"]
        self.assertAlmostEqual(energy, whole["energy_end_mwh_th"], places=11)
        self.assertAlmostEqual(loss, whole["standby_loss_mwh_th"], places=11)
        request = target_charge_request(10.0, 12.0, 1.0, self.c)
        restored = self.step(energy=10.0, requested=request)
        self.assertAlmostEqual(restored["energy_end_mwh_th"], 12.0, places=12)
        self.assertAlmostEqual(restored["energy_balance_error_mwh_th"], 0, places=12)

    def test_capacity_loss_crossing_physical_zero_is_flagged_not_clipped(self):
        self.c["capacity_loss_fraction_per_hour"] = 0.01
        out = self.step(energy=1.0, dt=10.0)
        self.assertAlmostEqual(out["energy_end_mwh_th"], -1.0)
        self.assertTrue(out["state_violation"])
        self.assertTrue(out["physical_energy_violation"])
        self.assertEqual(out["negative_energy_mwh_th"], 1.0)
        self.assertAlmostEqual(out["energy_balance_error_mwh_th"], 0)

    def test_efficiencies_and_energy_balance(self):
        self.c.update(charge_efficiency=0.9, discharge_efficiency=0.8,
                      hourly_retention_fraction=0.995)
        for requested in (-3.0, 0.0, 3.0):
            with self.subTest(requested=requested):
                out = self.step(requested=requested, dt=0.25)
                self.assertGreaterEqual(out["standby_loss_mwh_th"], 0)
                self.assertAlmostEqual(out["energy_balance_error_mwh_th"], 0, places=12)
                expected = (out["energy_start_mwh_th"] + out["charge_mw_th"] * 0.25
                            - out["discharge_mw_th"] * 0.25
                            - out["charge_conversion_loss_mwh_th"]
                            - out["discharge_conversion_loss_mwh_th"]
                            - out["standby_loss_mwh_th"])
                self.assertAlmostEqual(out["energy_end_mwh_th"], expected, places=12)

    def test_charge_headroom_includes_pump_heat(self):
        out = self.step(requested=-5.0, base=24.0, capacity=26.0, dt=1 / 12)
        self.assertAlmostEqual(out["charge_mw_th"], 2 / 1.02)
        self.assertAlmostEqual(out["chiller_mw_th"], 26.0)
        self.assertGreater(out["curtailed_mw_th"], 0)

    def test_no_headroom_no_charge(self):
        out = self.step(requested=-3.0, base=26.0, capacity=26.0)
        self.assertEqual(out["charge_mw_th"], 0)
        self.assertEqual(out["curtailed_mw_th"], 3.0)

    def test_discharge_cannot_make_chiller_negative(self):
        out = self.step(requested=5.0, base=2.0, dt=1 / 12)
        self.assertAlmostEqual(out["discharge_mw_th"], 2 / 0.98)
        self.assertAlmostEqual(out["chiller_mw_th"], 0)
        empty_base = self.step(requested=5.0, base=0.0, capacity=0.0)
        self.assertEqual(empty_base["discharge_mw_th"], 0)

    def test_mutual_exclusion(self):
        for requested in (-100.0, -1.0, 0.0, 1.0, 100.0):
            out = self.step(requested=requested)
            self.assertEqual(out["charge_mw_th"] * out["discharge_mw_th"], 0)

    def test_capacity_bounds_curtailment(self):
        upper = self.step(energy=18.0, requested=-5.0)
        lower = self.step(energy=2.0, requested=5.0)
        self.assertAlmostEqual(upper["energy_end_mwh_th"], 19.0)
        self.assertAlmostEqual(lower["energy_end_mwh_th"], 1.0)
        self.assertAlmostEqual(upper["curtailed_mw_th"], 4.0)
        self.assertAlmostEqual(lower["curtailed_mw_th"], 4.0)
        self.assertFalse(upper["state_violation"])
        self.assertFalse(lower["state_violation"])

    def test_equipment_power_limits(self):
        self.c.update(charge_limit_mw_th=1.2, discharge_limit_mw_th=1.5)
        self.assertEqual(self.step(requested=-5)["charge_mw_th"], 1.2)
        self.assertEqual(self.step(requested=5)["discharge_mw_th"], 1.5)

    def test_exact_constant_power_step_invariance(self):
        self.c.update(hourly_retention_fraction=0.99, charge_efficiency=0.91,
                      discharge_efficiency=0.87)
        for requested in (-1.0, 1.0):
            direct = self.step(requested=requested, dt=2.0)
            energy, loss = 10.0, 0.0
            for _ in range(24):
                sub = self.step(energy=energy, requested=requested, dt=1 / 12)
                self.assertEqual(sub["curtailed_mw_th"], 0)
                energy = sub["energy_end_mwh_th"]
                loss += sub["standby_loss_mwh_th"]
            self.assertAlmostEqual(energy, direct["energy_end_mwh_th"], places=11)
            self.assertAlmostEqual(loss, direct["standby_loss_mwh_th"], places=11)

    def test_idle_loss_below_min_is_reported_not_clipped(self):
        self.c["hourly_retention_fraction"] = 0.99
        out = self.step(energy=1.0, dt=1.0)
        self.assertAlmostEqual(out["energy_end_mwh_th"], 0.99)
        self.assertTrue(out["state_violation"])
        self.assertAlmostEqual(out["lower_bound_violation_mwh_th"], 0.01)
        self.assertAlmostEqual(out["energy_balance_error_mwh_th"], 0)

    def test_discharge_stops_when_idle_loss_exhausts_reserve(self):
        self.c["hourly_retention_fraction"] = 0.99
        out = self.step(energy=1.0, requested=1.0)
        self.assertEqual(out["discharge_mw_th"], 0)
        self.assertTrue(out["state_violation"])

    def test_target_recovery_uses_power_not_state_overwrite(self):
        self.c.update(hourly_retention_fraction=0.99, charge_efficiency=0.9)
        request = target_charge_request(10.0, 12.0, 1.0, self.c)
        self.assertLess(request, 0)
        out = self.step(energy=10.0, requested=request)
        self.assertAlmostEqual(out["energy_end_mwh_th"], 12.0, places=12)
        curtailed = self.step(energy=10.0, requested=request, base=25.9)
        self.assertLess(curtailed["energy_end_mwh_th"], 12.0)
        self.assertGreater(curtailed["curtailed_mw_th"], 0)
        self.assertEqual(target_charge_request(12.0, 10.0, 1.0, self.c), 0)

    def test_nonfinite_inputs_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for key in ("energy", "requested", "dt", "base", "capacity"):
                with self.subTest(value=value, key=key), self.assertRaises(ValueError):
                    self.step(**{key: value})
            for key in self.c:
                c = dict(self.c, **{key: value})
                with self.subTest(config_key=key, value=value), self.assertRaises(ValueError):
                    validate_tes(c)
            with self.assertRaises(ValueError):
                validate_tes(dict(self.c, capacity_loss_fraction_per_hour=value))

    def test_invalid_interval_baseline_and_initial_state(self):
        for args in ({"dt": 0}, {"dt": -1}, {"base": -1}, {"capacity": -1},
                     {"base": 27}, {"energy": 0.5}, {"energy": 20.0}, {"energy": -1}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.step(**args)

    def test_invalid_parameters_and_targets(self):
        for key, value in (("min_soc", -0.1), ("max_soc", 1.1),
                           ("energy_capacity_mwh_th", 0), ("charge_efficiency", 0),
                           ("discharge_efficiency", 1.1), ("hourly_retention_fraction", 0),
                           ("pump_mw_per_mw_th", 1), ("charge_limit_mw_th", -1),
                           ("capacity_loss_fraction_per_hour", -0.001)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_tes(dict(self.c, **{key: value}))
        c = dict(self.c)
        del c["charge_efficiency"]
        with self.assertRaises(ValueError):
            validate_tes(c)
        for target in (0, 20, float("nan")):
            with self.assertRaises(ValueError):
                target_charge_request(10.0, target, 1.0, self.c)


if __name__ == "__main__":
    unittest.main()
