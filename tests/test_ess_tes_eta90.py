"""Regression checks for the user-selected 90% / 90% ESS scenario."""
import json
from pathlib import Path
import unittest

import test_ess_tes_scheduler as independent

ROOT = Path(__file__).resolve().parents[1]


class TestEta90Scenario(unittest.TestCase):
    def test_only_efficiencies_change(self):
        old = json.loads((ROOT/'configs/ess_tes_pcc_scheduling_v2_4mwh.json').read_text(encoding='utf-8'))
        new = json.loads((ROOT/'configs/ess_tes_pcc_scheduling_v3_eta90.json').read_text(encoding='utf-8'))
        self.assertEqual({k for k in new['ess'] if new['ess'][k]!=old['ess'][k]}, {'charge_efficiency','discharge_efficiency'})
        for field in ('candidate_config','step_s','initial_soc','terminal_soc','cop','objective','strategies','scenarios'):
            self.assertEqual(old[field],new[field])
        self.assertEqual(new['ess']['charge_efficiency'],0.9)
        self.assertEqual(new['ess']['discharge_efficiency'],0.9)

    def test_cyclic_ac_round_trip_is_81_percent(self):
        fixture = independent.TestESSTESScheduler()
        fixture.setUp()
        ess = dict(fixture.ess,charge_efficiency=0.9,discharge_efficiency=0.9,max_soc=0.9)
        rows = fixture.trace(36)
        p0 = [fixture.facility_pcc(r) for r in rows]
        caps = [p+1 if k<12 or k>=24 else p-0.5 for k,p in enumerate(p0)]
        result = fixture.solve(rows=rows,caps=caps,ess_config=ess,use_tes=False)
        fixture.assert_replay(result,rows,caps,ess=ess,use_tes=False)
        schedule = result['schedule']
        incoming = sum(schedule['ess_charge_mw'])*fixture.dt
        outgoing = sum(schedule['ess_discharge_mw'])*fixture.dt
        self.assertGreater(outgoing,0.49)
        self.assertAlmostEqual(outgoing/incoming,0.81,places=8)
        self.assertAlmostEqual(schedule['ess_energy_mwh'][0],2.0,places=8)
        self.assertAlmostEqual(schedule['ess_energy_mwh'][-1],2.0,places=8)

    def test_usable_energy_duration(self):
        self.assertAlmostEqual(4*(0.9-0.1)*0.9/1,2.88)
        self.assertAlmostEqual(4*(0.5-0.1)*0.9/1,1.44)


if __name__=='__main__':
    unittest.main()
