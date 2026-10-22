"""Read saved output and independently recompute totals/formulas, no simulation refit."""
from __future__ import annotations
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/it_5min_operating_baseline_v1'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    summary=read(OUT/'summary.json'); design=summary['design']; data=np.load(OUT/'baseline_1s.npz')
    source=np.load(ROOT/'outputs/facility_arrival_scaling_v1/source_arrivals_1s.npz')
    five=pd.read_csv(OUT/'it_profiles_5min.csv'); errors={}
    pp=pd.read_csv(ROOT/'outputs/r3_high_rate_benchmark/train_validation_m3_parameters.csv').set_index('prompt_dataset')
    def power(r,d):
        p=pp.loc[d]
        return p.p0_kw+p.delta_p_kw/(1+np.exp(p.k*(p.x0-np.log1p(r))))
    expected=np.zeros(len(data['steady_mw']))
    for key,d in [('code','InstructCoder'),('conversation','mlperf-inference-llama2-data')]:
        counts=data['offered_'+key+'_requests']; n=design['instance_counts'][key]
        assert np.issubdtype(counts.dtype,np.integer)
        np.testing.assert_array_equal(counts.reshape(-1,60).sum(1),source[key].reshape(-1,60).sum(1)*design['request_multiplier'])
        np.testing.assert_allclose(counts/n,data[key+'_rate_per_instance_rps'],atol=0,rtol=0)
        np.testing.assert_array_equal(counts.reshape(-1,300).sum(1),five[key+'_offered_requests'])
        expected+=n*power(counts/n,d)/1000
    errors['m3_sum_max_abs_mw']=float(np.max(abs(expected-data['steady_mw'])))
    assert errors['m3_sum_max_abs_mw']<1e-12
    for i,seed in enumerate(summary['settings']['residual_seeds']):
        b=data['it_power_mw'][i].reshape(-1,300).mean(1)
        np.testing.assert_allclose(b,five[f'it_power_seed_{seed}_mw'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(data['it_power_mw'][i],data['steady_mw']+data['residual_mw'][i],atol=0,rtol=0)
        e=float(five[f'it_energy_seed_{seed}_mwh'].sum())
        assert abs(e-data['it_power_mw'][i].sum()/3600)<1e-8
        assert abs(e-summary['baseline'][i]['energy_mwh'])<1e-8
    t=pd.to_datetime(five.interval_start_utc,utc=True)
    assert str(t.iloc[0])==str(pd.Timestamp(summary['start_utc']))
    assert (t.diff().iloc[1:].dt.total_seconds()==300).all()
    assert t.iloc[-1]+pd.Timedelta(seconds=300)==pd.Timestamp(summary['end_exclusive_utc'])
    carry=pd.read_csv(OUT/'carryover_profiles_1s.csv')
    max_metric_error=0.
    for row in summary['sensitivity']:
        x=carry[carry.case.eq(row['case'])]
        names={'full_vs_original':('facility_original_mw','facility_stateful_bridge_mw'),
            'memory_only':('facility_memoryless_bridge_mw','facility_stateful_bridge_mw'),
            'static_bridge_offset':('facility_original_mw','facility_memoryless_bridge_mw')}
        a,b=names[row['comparison']]; offset=row['offset_s']; length=((len(x)-offset)//300)*300
        aa=x[a].to_numpy()[offset:offset+length].reshape(-1,300).mean(1)
        bb=x[b].to_numpy()[offset:offset+length].reshape(-1,300).mean(1)
        delta=bb-aa
        metrics={'mae_mw':float(abs(delta).mean()),'max_abs_difference_mw':float(abs(delta).max()),
            'difference_energy_mwh':float(delta.sum()/12),'mean_difference_mw':float(delta.mean())}
        for k,v in metrics.items(): max_metric_error=max(max_metric_error,abs(row[k]-v))
    assert max_metric_error<1e-11
    state=read(OUT/'carryover_state_checks.json')
    for case,services in state.items():
        for service,x in services.items():
            f=x['final_state']
            assert f['arrivals']==f['completions']+f['running']+f['waiting']
            assert x['conservation_max_error_requests']==0
            assert abs(x['generated_arrivals']-x['expected_arrivals_with_warmup'])<1
    old=read(OUT/'preservation_hashes.json')
    for path,value in old.items():assert digest(Path(path))==value,path
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ('href','src') and not v.startswith(('http','#')):self.links.append(v)
    parser=Links();parser.feed((OUT/'report_ko.html').read_text(encoding='utf-8'))
    for path in parser.links:assert (OUT/path).exists(),path
    result={'all_passed':True,'complete_5min_bins':len(five),'seconds_per_service':len(expected),
        'residual_realizations':len(summary['settings']['residual_seeds']),
        'exact_request_total_checks':'each minute and each five-minute interval for both services',
        'protected_files_unchanged':len(old),'local_report_links_checked':len(parser.links),
        'recomputed_sensitivity_rows':len(summary['sensitivity']),
        'max_metric_recalculation_difference':max_metric_error,**errors,
        'verification_is_real_world_validation':False}
    (OUT/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
