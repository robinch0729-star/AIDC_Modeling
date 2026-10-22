import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from aidc_facility.it_operating_baseline import (m3,block_mean,choose_design,locked_coefficients,
    aggregate_innovation,sum_residual_seconds,regular_variable_arrivals,carryover_bridge,paired_metrics,PARAM_COLUMNS)

P={'p0_kw':2.,'delta_p_kw':1.,'k':2.,'x0':2.}


class TestOperatingBaseline(unittest.TestCase):
    def test_m3_monotone_bounded(self):
        y=m3(np.linspace(0,100,1000),P)
        self.assertTrue(np.all(np.diff(y)>0)); self.assertTrue(np.all((y>=2)&(y<=3)))
        with self.assertRaises(ValueError): m3([-1],P)

    def test_aggregation_energy_and_offset(self):
        x=np.arange(900.)
        self.assertAlmostEqual(block_mean(x).sum()/12,x.sum()/3600)
        np.testing.assert_allclose(block_mean(x,300,150),[299.5,599.5])
        with self.assertRaises(ValueError): block_mean(x,300,300)

    def test_integer_design_no_output_scaling(self):
        x={'code':np.full(600,20,dtype=int),'conversation':np.full(600,30,dtype=int)}
        d=choose_design(x,{'code':P,'conversation':P},.12,max_multiplier=1000)
        self.assertIsInstance(d['request_multiplier'],int)
        n=d['instance_counts']; k=d['request_multiplier']
        self.assertTrue(all(isinstance(v,int) for v in n.values()))
        expected=sum(n[s]*m3(x[s].mean()*k/n[s],P) for s in x)/1000
        self.assertAlmostEqual(expected,d['design_expected_mean_mw'])
        self.assertAlmostEqual(sum(d['source_mix'].values()),1)

    def test_neighbour_weight_exact(self):
        rows=[]
        for r in range(1,11):
            row={name:.01*r for name in PARAM_COLUMNS}
            row.update(prompt_dataset='a',request_rate=r,phi=.1*r,jump_evidence=.5)
            rows.append(row)
        lib=pd.DataFrame(rows)
        c=locked_coefficients(lib,'a',np.array([4.5,0.,1.]))
        dist=np.abs(np.arange(1,11)-4.5); idx=np.argsort(dist,kind='stable')[:8]
        weights=1/np.maximum(dist[idx],.5); weights/=weights.sum()
        self.assertAlmostEqual(c['phi'][0],np.sum((idx+1)*.1*weights))
        np.testing.assert_allclose(c['phi'][1],c['phi'][2])
        np.testing.assert_allclose(c['prob'].sum(1),1)
        np.testing.assert_allclose((c['prob']*c['mean']).sum(1),0,atol=1e-15)

    def test_sum_mixture_moments(self):
        size=100000; n=7
        p=np.tile([.3,.2,.4,.1],(size,1)); m=np.tile([0,-2,.5,2],(size,1)); sd=np.tile([.2,.4,.1,.8],(size,1))
        z=aggregate_innovation(np.random.default_rng(9),n,p,m,sd)
        mean=(p[0]*m[0]).sum(); var=(p[0]*(sd[0]**2+m[0]**2)).sum()-mean**2
        self.assertLess(abs(z.mean()-n*mean),.04)
        self.assertLess(abs(z.var()/(n*var)-1),.02)

    def test_deterministic_components(self):
        p=np.tile([0,0,0,1.],(4,1)); m=np.tile([0,0,0,2.],(4,1)); sd=np.zeros_like(p)
        np.testing.assert_array_equal(aggregate_innovation(np.random.default_rng(3),5,p,m,sd),10)

    def test_native_dt_and_seed(self):
        c={'phi':np.full(40,.9),'prob':np.tile([1,0,0,0],(40,1)),
           'mean':np.zeros((40,4)),'std':np.full((40,4),.1)}
        a,_=sum_residual_seconds(c,10,4,burn_seconds=1)
        b,_=sum_residual_seconds(c,10,4,burn_seconds=1)
        np.testing.assert_array_equal(a,b)
        with self.assertRaises(ValueError): sum_residual_seconds(c,10,4,native_dt=1)

    def test_variable_arrivals_causal_with_zeros(self):
        r=np.array([0.,2.,0.,3.])
        a=regular_variable_arrivals(r)
        np.testing.assert_allclose(a,[1.25,1.75,3+1/6,3.5,3+5/6])
        self.assertTrue(np.all(np.diff(a)>0))

    def test_bridge_request_balance_and_memory(self):
        params={'concurrency_limit_hypothesis':4,'state_curve_running_knots':[0,1,4],
                'state_curve_rate_knots':[0,1,2]}
        r=np.r_[np.full(15,5.),np.zeros(40)]
        values,check=carryover_bridge(r,params,P,warmup_s=0,sample_dt=.5)
        self.assertEqual(check['conservation_max_error_requests'],0)
        self.assertEqual(check['final_state']['unfinished'],0)
        self.assertGreater(values['waiting_mean'].max(),0)
        self.assertGreater(values['stateful_bridge_kw'][15],values['original_kw'][15])
        self.assertEqual(check['final_state']['arrivals'],check['final_state']['completions'])

    def test_pair_metrics_direction_energy(self):
        m=paired_metrics(np.ones(600)*10,np.ones(600)*11)
        self.assertAlmostEqual(m['mae_mw'],1)
        self.assertAlmostEqual(m['difference_energy_mwh'],1/6)


if __name__=='__main__': unittest.main()
