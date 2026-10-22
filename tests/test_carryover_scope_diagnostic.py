import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from aidc_facility.carryover_scope_diagnostic import (exposure_rates,size_for_mean_rate,
    block_rates,window_summary,selected_errors,waiting_onset,interval_diagnostics,replay_diagnostic)
from aidc_facility.carryover_replay import simulate


def samples():
    t=np.arange(6)*10.
    return pd.DataFrame({"timestamp":pd.Timestamp('2026-01-01')+pd.to_timedelta(t,unit='s'),
        "relative_run_s":60+t,"replay_s":t,"running_requests":10.,"waiting_requests":t,
        "prompt_tokens_per_s":[999,10,20,30,40,50],"generation_tokens_per_s":[999,20,40,60,80,100],
        "source_line":np.arange(6)+1,"gpu_kv_cache_pct":5.,"prefix_cache_hit_pct":10.})


P={"concurrency_limit_hypothesis":20,"state_curve_running_knots":[0,10,20],"state_curve_rate_knots":[0,5,10]}


class ScopeDiagnosticTests(unittest.TestCase):
    def test_bands_exhaustive_at_boundaries(self):
        d=exposure_rates([0,1.9,2,30,30.1,59.9,60,100,100.1])
        self.assertAlmostEqual(sum(d.values()),1)
        self.assertAlmostEqual(d['2_to_30'],2/9)
        self.assertAlmostEqual(d['60_to_100'],2/9)

    def test_invalid_exposure_input(self):
        for v in ([],[-1],[np.nan]):
            with self.assertRaises(ValueError): exposure_rates(v)

    def test_integer_allocation_does_not_exceed_target_mean(self):
        n=size_for_mean_rate(101,20)
        self.assertEqual(n,6); self.assertLessEqual(101/n,20)

    def test_invalid_sizing(self):
        with self.assertRaises(ValueError): size_for_mean_rate(0,20)

    def test_block_count_conservation(self):
        x=np.arange(12)
        r=block_rates(x,3)
        self.assertEqual(sum(r)*3,sum(x))

    def test_partial_block_not_silently_used(self):
        with self.assertRaises(ValueError): block_rates(np.arange(10),3)

    def test_aligned_token_mean_omits_first_interval(self):
        d=window_summary(samples(),5,P,60,120)
        self.assertEqual(d['generation_tokens_per_s_aligned_mean'],60)
        self.assertEqual(d['prompt_tokens_per_s_aligned_mean'],30)
        self.assertAlmostEqual(d['conditional_departure_rps'],4)

    def test_balance_proxy_not_token_rate(self):
        r=interval_diagnostics(samples(),5,P)
        self.assertTrue(all(x['conditional_departure_rps']==4 for x in r))
        self.assertEqual(r[0]['generation_tokens_per_s'],20)

    def test_initial_point_excluded(self):
        o=samples(); p=pd.DataFrame({'time_s':o.replay_s,'running':10,'waiting':o.waiting_requests})
        p.loc[0,'running']=1000
        self.assertEqual(selected_errors(o,p)['running_rmse'],0)

    def test_onset_is_sampled_time_not_interpolation(self):
        o=samples(); p=pd.DataFrame({'waiting':[0,0,1,2,3,4]})
        x=waiting_onset(o,p)
        self.assertEqual(x['observed_first_sampled_waiting_run_s'],70)
        self.assertEqual(x['predicted_first_sampled_waiting_run_s'],80)

    def test_diagnostic_engine_matches_frozen_model(self):
        o=samples(); o.waiting_requests=0
        a=replay_diagnostic(o,5,P)
        b=simulate('S1_state_dependent_lps',P,5,o.replay_s.to_numpy(),10,0)
        np.testing.assert_array_equal(a[['running','waiting']],b[['running','waiting']])
        self.assertEqual(a.initial_running_jobs_left_in_model.iloc[-1],0)
        self.assertEqual(abs(a.conservation_error).max(),0)

if __name__=='__main__': unittest.main()
