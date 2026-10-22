"""Regression checks for numeric JSON cap values in the capacity comparison."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_ess_tes_4mwh import make_cap


class FloatingPCCCapTests(unittest.TestCase):
    def setUp(self):
        self.power = np.array([22.6, 23.1, 24.2, 25.3])
        self.hours = np.arange(4, dtype=float)
        self.coefficients = {'reduction_mwe_per_mwth': .2139889898989899}

    def test_integer_constant_preserves_float_dtype(self):
        cap = make_cap({'type': 'constant', 'cap_mw': 26}, self.power,
                       self.hours, self.coefficients)
        self.assertEqual(cap.dtype.kind, 'f')
        np.testing.assert_array_equal(cap, np.full(4, 26.))

    def test_integer_outside_cap_does_not_truncate_stress(self):
        scenario = {'type': 'energy_stress', 'outside_cap_mw': 27,
                    'start_h': 1, 'end_h': 3, 'thermal_discharge_required_mw_th': 2}
        cap = make_cap(scenario, self.power, self.hours, self.coefficients)
        expected = np.array([27., 23.1-2*self.coefficients['reduction_mwe_per_mwth'],
                             24.2-2*self.coefficients['reduction_mwe_per_mwth'], 27.])
        np.testing.assert_array_equal(cap, expected)
        self.assertEqual(cap.dtype.kind, 'f')

    def test_integer_and_float_json_values_produce_identical_caps(self):
        integer = {'type': 'time_window', 'outside_cap_mw': 27,
                   'inside_cap_mw': 25.5, 'start_h': 1, 'end_h': 3}
        floating = dict(integer, outside_cap_mw=27.)
        a = make_cap(integer, self.power, self.hours, self.coefficients)
        b = make_cap(floating, self.power, self.hours, self.coefficients)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, np.array([27., 25.5, 25.5, 27.]))

    def test_cap_builder_does_not_mutate_input(self):
        scenario = {'type': 'constant', 'cap_mw': 26}
        before = deepcopy(scenario)
        power = self.power.copy()
        make_cap(scenario, self.power, self.hours, self.coefficients)
        self.assertEqual(scenario, before)
        self.assertIsInstance(scenario['cap_mw'], int)
        np.testing.assert_array_equal(power, self.power)


if __name__=='__main__':
    unittest.main()
