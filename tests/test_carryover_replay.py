import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from aidc_facility.carryover_replay import StateReplay, regular_arrivals, isotonic, simulate


class ReplayTests(unittest.TestCase):
    def test_future_arrival_not_served_early(self):
        e=StateReplay([10.],0,0,1,lifetime=2.)
        self.assertEqual(e.advance(9.)["unfinished"],0)
        self.assertEqual(e.advance(10.)["running"],1)
        self.assertEqual(e.advance(11.)["completions"],0)
        self.assertEqual(e.advance(12.)["completions"],1)

    def test_fifo_limit_and_conservation(self):
        e=StateReplay([1.,1.,1.],0,0,1,lifetime=2.)
        r=e.advance(1.); self.assertEqual((r["running"],r["waiting"]),(1,2))
        self.assertEqual(e.advance(3.)["waiting"],1)
        self.assertEqual(e.advance(7.)["unfinished"],0)
        self.assertEqual(e.snapshot()["conservation_error"],0)

    def test_ps_equal_work_shares_capacity(self):
        e=StateReplay([1.,1.],0,0,2,running_knots=[0,1,2],rate_knots=[0,1,1])
        self.assertEqual(e.advance(2.9)["running"],2)
        self.assertEqual(e.advance(3.)["completions"],2)

    def test_ps_state_dependent_capacity(self):
        e=StateReplay([1.,1.],0,0,2,running_knots=[0,1,2],rate_knots=[0,1,2])
        self.assertEqual(e.advance(2.)["completions"],2)

    def test_initial_state_then_queued_admission(self):
        e=StateReplay([],1,3,2,lifetime=2.)
        self.assertEqual(e.advance(0.)["running"],1)
        self.assertEqual(e.advance(.1)["running"],2)
        self.assertEqual(e.advance(20.)["completions"],4)

    def test_output_sampling_does_not_reset_state(self):
        args=([.3,1.1,2.,4.2],2,3,4)
        a=StateReplay(*args,lifetime=2.)
        b=StateReplay(*args,lifetime=2.)
        for t in np.arange(.1,10.01,.1): a.advance(t)
        rb=b.advance(10.)
        for key in ("running","waiting","arrivals","completions","unfinished"):
            self.assertEqual(a.snapshot()[key],rb[key])

    def test_no_reset_at_300_seconds(self):
        e=StateReplay([299.],0,0,1,lifetime=10.)
        self.assertEqual(e.advance(300.)["running"],1)
        self.assertEqual(e.advance(309.)["completions"],1)

    def test_isotonic_known_solution(self):
        np.testing.assert_allclose(isotonic([0,3,1,4]),[0,2,2,4])

    def test_regular_arrivals_count_and_phase(self):
        a=regular_arrivals(20,10)
        self.assertEqual(len(a),200)
        self.assertAlmostEqual(a[0],.025)

    def test_invalid_arrivals(self):
        for a in ([0],[-1],[2,1],[np.nan]):
            with self.assertRaises(ValueError): StateReplay(a,0,0,1,lifetime=1)

    def test_invalid_initial_state(self):
        with self.assertRaises(ValueError): StateReplay([],3,0,2,lifetime=1)

    def test_refuse_backward_time(self):
        e=StateReplay([],0,0,1,lifetime=1); e.advance(2)
        with self.assertRaises(ValueError): e.advance(1)

    def test_irregular_and_dense_sampling_agree_for_ps(self):
        a=StateReplay(regular_arrivals(15,20),20,5,32,running_knots=[0,16,32],rate_knots=[0,8,12])
        b=StateReplay(regular_arrivals(15,20),20,5,32,running_knots=[0,16,32],rate_knots=[0,8,12])
        for t in np.arange(.1,20.01,.1): a.advance(t)
        b.advance(20)
        self.assertEqual(a.snapshot()["unfinished"],b.snapshot()["unfinished"])


if __name__=="__main__": unittest.main()
