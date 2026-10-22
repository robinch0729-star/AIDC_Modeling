import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aidc_facility.pcc_cooling_tes import (affine_thermal_step,tank_capacity,tank_step,
    simulate,summarize,aggregate,validate)


class TestPCCCoolingTES(unittest.TestCase):
    def setUp(self):
        self.c=json.loads((ROOT/'configs/pcc_cooling_tes_candidate_v1.json').read_text())

    def test_unit_conversion(self):
        self.assertAlmostEqual(tank_capacity(self.c['tes']),3.4833333333333334)

    def test_adiabatic_heat_balance(self):
        end,heat=affine_thermal_step(24,30,2,1,0,.5)
        self.assertEqual(end,25); self.assertEqual(heat,0)

    def test_exact_equilibrium(self):
        end,_=affine_thermal_step(24,30,-.12,1,.02,1)
        self.assertAlmostEqual(end,24)

    def test_tank_charge_cools(self):
        x=tank_step(8,-2,10/3600,self.c['tes'],18,26)
        self.assertLess(x['end'],8); self.assertGreater(x['flow'],0)

    def test_tank_discharge_warms(self):
        x=tank_step(8,2,10/3600,self.c['tes'],18,26)
        self.assertGreater(x['end'],8)

    def test_hot_tank_cannot_supply(self):
        x=tank_step(10.1,4,10/3600,self.c['tes'],18,26)
        self.assertLess(x['net'],1e-7)

    def test_no_chiller_headroom_for_charge(self):
        x=tank_step(8,-4,10/3600,self.c['tes'],26,26)
        self.assertLess(abs(x['net']),1e-7)

    def test_flow_and_temperature_limit(self):
        t=self.c['tes']; x=tank_step(9.999,5,300/3600,t,18,26)
        self.assertLessEqual(x['flow'],t['maximum_flow_kg_s']+1e-6)
        # Ambient heat alone may push an inactive tank above the supply ceiling.
        # A supply constraint must stop discharge, not magically cool the tank.
        if x['net']>1e-7: self.assertLessEqual(x['supply'],10+1e-7)
        else: self.assertLess(x['net'],1e-7)
        self.assertGreater(x['curtailed'],0)

    def test_hx_approach_not_just_tank_temperature(self):
        # Tank <10 C is insufficient: with epsilon=.95 and return=12 C,
        # the warmest tank that can produce <=10 C supply is 9.8947368 C.
        x=tank_step(9.95,3,10/3600,self.c['tes'],18,26)
        self.assertLess(x['net'],1e-7)

    def test_full_balance_and_aggregation(self):
        f=simulate(np.full(360,17.4),self.c,True,commands=np.full(360,2.))
        s=summarize(f,self.c,True)
        self.assertTrue(s['feasible_within_declared_checks'])
        for error in s['max_abs_balances'].values(): self.assertLess(error,1e-8)
        a=aggregate(f,self.c)
        self.assertAlmostEqual(f.pcc_mw.sum()*10,a.pcc_mw.sum()*300,places=7)

    def test_insufficient_chiller_is_reported_not_temperature_clipped(self):
        c=copy.deepcopy(self.c); c['plant']['chiller_capacity_mw_th']=10
        f=simulate(np.full(360,20.),c)
        self.assertGreater(f.room_end_c.max(),27)
        self.assertGreater(summarize(f,c,False)['constraints']['room_temperature_violations'],0)

    def test_no_tes_same_input_unchanged(self):
        x=np.linspace(16,19,360); f=simulate(x,self.c)
        np.testing.assert_array_equal(x,f.it_mw)
        self.assertEqual(f.tes_net_mw_th.abs().max(),0)

    def test_invalid_efficiency(self):
        self.c['electrical']['ups_efficiency']=1.2
        with self.assertRaises(ValueError): validate(self.c)

    def test_nothing_is_named_measured_site_validation(self):
        self.assertIn('not_selected',self.c['status'])


if __name__=='__main__': unittest.main()
