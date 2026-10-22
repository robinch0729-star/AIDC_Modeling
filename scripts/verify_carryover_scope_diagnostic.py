"""Independent checks of phase 4-3 arithmetic against stored inputs."""
from pathlib import Path
import json
import re
import sys
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aidc_facility.nlr_processing_diagnostic import sha256
from run_carryover_replay import dump


def main():
    out=ROOT/'outputs/carryover_scope_diagnostic_v1'
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    protocol=read(out/'protocol.json')
    for p,h in read(out/'preservation_hashes.json').items(): assert sha256(Path(p))==h
    for x in read(out/'source_manifest.json'): assert sha256(Path(x['path']))==x['sha256']
    raw=pd.read_csv(ROOT/'outputs/nlr_processing_diagnostic_v1/raw_server_metrics.csv',parse_dates=['timestamp'])
    runs=pd.read_csv(ROOT/'outputs/nlr_transition_audit_v1/inference_online_rate_llama3_70b_runs.csv',dtype={'id':str}).set_index('id')
    maximum=0.
    windows=read(out/'window_diagnostics.json')
    for row in windows:
        start=pd.Timestamp(runs.loc[row['run_id'],'run_start']); left,right=protocol['windows_relative_to_run_s'][row['window']]
        elapsed=(raw.timestamp-start).dt.total_seconds()
        s=raw[(elapsed>=left)&(elapsed<right)]
        t=(s.timestamp-s.timestamp.iloc[0]).dt.total_seconds().to_numpy()
        b=(s.running_requests+s.waiting_requests).to_numpy()
        departure=row['rate']-np.polyfit(t,b,1)[0]
        tokens=np.sum(s.generation_tokens_per_s.to_numpy()[1:]*np.diff(t))/(t[-1]-t[0])
        average_R=np.trapezoid(s.running_requests.to_numpy(),t)/(t[-1]-t[0])
        for actual,key in ((departure,'conditional_departure_rps'),(tokens,'generation_tokens_per_s_aligned_mean'),(average_R,'running_mean')):
            maximum=max(maximum,abs(actual-row[key]))
            assert np.isclose(actual,row[key],rtol=1e-12,atol=1e-9)
    scores=pd.DataFrame(read(out/'initial_work_sensitivity.json'))
    trace_count=0
    for row in read(out/'focus_plot_data.json'):
        for i,pred in enumerate(row['initial_work_traces']):
            p=pd.DataFrame(pred); assert (p.unfinished==p.unfinished.iloc[0]+p.arrivals-p.completions).all()
            power=protocol['initial_work_powers'][i//3]; phase=protocol['arrival_phases'][i%3]
            s=scores[(scores.service==row['service'])&(scores.rate==row['rate'])&scores.initial_work_power.eq(power)&scores.arrival_phase.eq(phase)].iloc[0]
            for name in ('running','waiting'):
                err=p[name].to_numpy()[1:]-np.asarray(row['observed_'+name])[1:]
                assert np.isclose(np.sqrt(np.mean(err**2)),s[name+'_rmse'],rtol=1e-12)
            trace_count+=1
    arrivals=ROOT/'outputs/facility_arrival_scaling_v1'
    names={'scaled_replay':'facility_arrivals_replay_K1000_seed20261005.npz',
        'conditional_60s':'facility_arrivals_conditional_60s_K1000_seed20261005.npz',
        'conditional_300s':'facility_arrivals_conditional_300s_K1000_seed20261005.npz'}
    exposure=read(out/'facility_exposure.json')
    cached={k:dict(np.load(arrivals/name,allow_pickle=False)) for k,name in names.items()}
    for row in exposure:
        x=cached[row['method']][row['service']]; seconds=row['bin_seconds']; n=row['illustrative_instances']
        rate=x.reshape(-1,seconds).sum(axis=1)/seconds/n
        boundaries=[(rate<2),((rate>=2)&(rate<=30)),((rate>30)&(rate<60)),((rate>=60)&(rate<=100)),(rate>100)]
        for mask,key in zip(boundaries,['below_2','2_to_30','over_30_below_60','60_to_100','over_100']):
            assert np.isclose(np.count_nonzero(mask)/len(rate),row[key],atol=1e-15)
        assert n==int(np.ceil(x.sum()/len(x)/row['illustrative_mean_target_rps']))
    html=(out/'report_ko.html').read_text(encoding='utf-8')
    for link in re.findall(r'(?:src|href)="([^"]+)"',html):
        if not link.startswith(('http:','https:','#')): assert (out/link).exists(),link
    loader=unittest.TestLoader(); suite=unittest.TestSuite()
    for pattern in ('test_*carryover*.py','test_nlr_processing_diagnostic.py'):
        suite.addTests(loader.discover(str(ROOT/'tests'),pattern=pattern))
    count=suite.countTestCases(); result=unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful()
    output={'all_checks_passed':True,'window_summaries_independently_recalculated':len(windows),
        'max_scalar_recalculation_difference':maximum,'initial_sensitivity_trajectories_rechecked':trace_count,
        'facility_exposure_rows_independently_recalculated':len(exposure),'local_report_links_exist':True,
        'protected_files_match':True,'source_hashes_match':True,'unit_tests_passed':count,
        'verifier_sha256':sha256(Path(__file__))}
    dump(out/'independent_verification.json',output); print(json.dumps(output,indent=2))


if __name__=='__main__': main()
