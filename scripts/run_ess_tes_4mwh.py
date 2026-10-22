"""Fixed-power 2 versus 4 MWh ESS comparison; preserve all v1 outputs."""
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import scipy

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from aidc_facility.tes_flexibility import marginal_pcc_coefficients, pcc_from_thermal
from run_ess_tes_pcc_scheduling import solve
from run_tes_pcc_scheduling import read, dump, sha, cap_metrics, make_cap as legacy_make_cap

CONFIG = ROOT/'configs/ess_tes_pcc_scheduling_v2_4mwh.json'
OUT = ROOT/'outputs/ess_tes_pcc_scheduling_v2_4mwh'


def make_cap(case, p0, hours, coefficients):
    # Legacy helper inherits dtype from np.full's fill value. JSON permits
    # 27 and 27.0 for the same value, so explicitly retain floating cap arrays
    # before assigning noninteger stress levels. Do not change the v1 helper.
    spec = dict(case)
    for key in ('cap_mw', 'outside_cap_mw', 'inside_cap_mw'):
        if key in spec:
            spec[key] = float(spec[key])
    return legacy_make_cap(spec, p0, hours, coefficients)


def main():
    cfg = read(CONFIG)
    old_cfg = read(ROOT/'configs/ess_tes_pcc_scheduling_v1.json')
    candidate = read(ROOT/cfg['candidate_config'])
    legacy = read(ROOT/candidate['legacy_config'])
    tes = candidate['tes']
    old = read(ROOT/cfg['comparison_baseline'])
    changes = [k for k in cfg['ess'] if cfg['ess'][k] != old_cfg['ess'][k]]
    common_keys = ('candidate_config', 'step_s', 'initial_soc', 'terminal_soc', 'cop',
                   'objective', 'time_limit_s', 'strategies', 'scenarios')
    common_unchanged = all(cfg[k] == old_cfg[k] for k in common_keys)
    if changes != ['energy_capacity_mwh'] or not common_unchanged:
        raise AssertionError('Only ESS energy capacity may change in the physical comparison')
    if cfg['ess']['energy_capacity_mwh'] != 4 or old_cfg['ess']['energy_capacity_mwh'] != 2:
        raise AssertionError('Expected fixed-power 2 versus 4 MWh comparison')

    protected = set()
    for name in ('ess_tes_pcc_scheduling_v1', 'tes_pcc_scheduling_v1',
                 'tes_flexibility_assessment_v1', 'pcc_tes_operating_candidate_v1',
                 'pcc_cooling_tes_candidate_v1', 'it_pcc_boundary_audit_v1',
                 'pcc_capacity_cop_review_v1', 'pcc_temperature_conditions_v1'):
        protected.update(p for p in (ROOT/'outputs'/name).rglob('*') if p.is_file())
    for name in ('configs/ess_tes_pcc_scheduling_v1.json', 'configs/tes_pcc_scheduling_v1.json',
                 cfg['candidate_config'], candidate['legacy_config'],
                 'src/aidc_facility/ess_tes_scheduler.py', 'src/aidc_facility/tes_scheduler.py',
                 'src/aidc_facility/tes_operating.py', 'src/aidc_facility/tes_flexibility.py',
                 'scripts/run_ess_tes_pcc_scheduling.py', 'scripts/run_tes_pcc_scheduling.py',
                 'tests/test_ess_tes_scheduler.py', 'tests/test_tes_scheduler.py'):
        path = ROOT/name
        if path.exists():
            protected.add(path)
    before = {str(p.relative_to(ROOT)): sha(p) for p in sorted(protected)}
    cols = read(ROOT/candidate['baseline_columns'])
    original = np.load(ROOT/candidate['baseline_profiles'])['no_tes']
    block = cfg['step_s']//legacy['internal_step_s']
    if cfg['step_s'] % legacy['internal_step_s'] or len(original) % block:
        raise ValueError('Complete aggregation bins required')
    base = original.reshape(-1, block, len(cols)).mean(1)
    rows = [dict(zip(cols, map(float, row))) for row in base]
    dt = cfg['step_s']/3600
    hours = np.arange(len(rows))*dt
    p0 = np.array([pcc_from_thermal(r, 0., 0., legacy, cfg['cop'], tes['pump_mw_per_mw_th'])['pcc_mw'] for r in rows])
    coeff = marginal_pcc_coefficients(legacy, cfg['cop'], tes['pump_mw_per_mw_th'])
    with np.load(ROOT/'outputs/ess_tes_pcc_scheduling_v1/schedules_5min.npz') as old_arrays:
        cap_differences = {case['id']: float(np.max(np.abs(
            make_cap(case, p0, hours, coeff)-old_arrays[case['id']+'_cap_mw'])))
            for case in cfg['scenarios']}
    identical_caps = all(error<1e-12 for error in cap_differences.values())
    if not identical_caps:
        raise AssertionError('PCC cap arrays must match v1 exactly: '+str(cap_differences))
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {'time_h': hours, 'no_storage_pcc_mw': p0}
    solutions, records, summaries = {}, {}, []

    def save(name, result):
        if name in records:
            raise AssertionError('Duplicate solution: '+name)
        records[name] = result
        dump(OUT/name, result)

    for case in cfg['scenarios']:
        cap = make_cap(case, p0, hours, coeff)
        arrays[case['id']+'_cap_mw'] = cap
        for strategy in cfg['strategies']:
            key = (case['id'], strategy['id'])
            print('Solving '+'/'.join(key), flush=True)
            result = solve(rows, cap, tes, legacy, cfg, strategy)
            solutions[key] = result
            save('_'.join(key)+'_solution.json', result)
            item = {'case_id': case['id'], 'case_label_ko': case['label_ko'],
                    'strategy_id': strategy['id'], 'strategy_label_ko': strategy['label_ko'],
                    'status': result['status'], 'feasible': result['feasible'], 'optimal': result['optimal'],
                    'solver': result['solver'], 'no_storage': cap_metrics(p0, cap, dt)}
            if result['feasible']:
                s = result['schedule']
                item['operating_metrics'] = cap_metrics(s['pcc_mw'], cap, dt)
                item['kernel_metrics'] = result['metrics']
                item['ess_discharge_mwh'] = float(np.sum(s['ess_discharge_mw'])*dt)
                item['tes_discharge_mwh_th'] = float(np.sum(s['tes_discharge_mw_th'])*dt)
                delta = coeff['increase_mwe_per_mwth']*np.asarray(s['tes_charge_mw_th'])-coeff['reduction_mwe_per_mwth']*np.asarray(s['tes_discharge_mw_th'])+np.asarray(s['ess_charge_mw'])-np.asarray(s['ess_discharge_mw'])
                p10 = original[:, cols.index('pcc_mw')]+np.repeat(delta, block)
                q10 = original[:, cols.index('chiller_mw_th')]+np.repeat((1+tes['pump_mw_per_mw_th'])*np.asarray(s['tes_charge_mw_th'])-(1-tes['pump_mw_per_mw_th'])*np.asarray(s['tes_discharge_mw_th']), block)
                item['held_10s'] = cap_metrics(p10, np.repeat(cap, block), legacy['internal_step_s']/3600)
                item['held_10s']['minimum_chiller_headroom_mw_th'] = float(legacy['plant']['chiller_capacity_mw_th']-q10.max())
                item['held_10s']['minimum_chiller_output_mw_th'] = float(q10.min())
                item['held_10s']['reaggregation_error_mw'] = float(np.max(np.abs(p10.reshape(-1, block).mean(1)-s['pcc_mw'])))
                for name, value in s.items():
                    arrays['_'.join(key)+'_'+name] = np.asarray(value)
            summaries.append(item)
            print(json.dumps({'case': case['id'], 'strategy': strategy['id'], 'status': result['status'],
                              'energy_mwh': item.get('operating_metrics', {}).get('pcc_energy_mwh')}, ensure_ascii=False), flush=True)

    joint_strategy = next(s for s in cfg['strategies'] if s['id']=='joint')
    matched = []
    for cid in ('constant_26', 'constant_25_8', 'constant_25_5'):
        case = next(c for c in cfg['scenarios'] if c['id']==cid)
        cap = make_cap(case, p0, hours, coeff)
        idle = solve(rows, cap, dict(tes, discharge_limit_mw_th=0.), legacy, cfg, joint_strategy)
        save(cid+'_ess_with_tes_idle_solution.json', idle)
        joint = solutions[(cid, 'joint')]
        delta = joint['metrics']['pcc_energy_mwh']-idle['metrics']['pcc_energy_mwh'] if joint['feasible'] and idle['feasible'] else None
        matched.append({'case_label_ko': case['label_ko'], 'idle_status': idle['status'],
                        'joint_status': joint['status'], 'joint_minus_idle_mwh': delta})
        print('Matched hardware '+cid+': '+idle['status'], flush=True)

    comparison = []
    for item in summaries:
        previous = next(r for r in old['results'] if r['case_id']==item['case_id'] and r['strategy_id']==item['strategy_id'])
        entry = {'case_id': item['case_id'], 'case_label_ko': item['case_label_ko'],
                 'strategy_id': item['strategy_id'], 'strategy_label_ko': item['strategy_label_ko'],
                 'old_status': previous['status'], 'new_status': item['status'],
                 'old_feasible': previous['feasible'], 'new_feasible': item['feasible']}
        for field in ('pcc_energy_mwh', 'ess_charge_energy_mwh', 'ess_discharge_energy_mwh',
                      'tes_charge_energy_mwh_th', 'tes_discharge_energy_mwh_th',
                      'ess_minimum_soc', 'ess_maximum_soc', 'pcc_peak_mw'):
            a = previous.get('kernel_metrics', {}).get(field)
            b = item.get('kernel_metrics', {}).get(field)
            entry['old_'+field], entry['new_'+field] = a, b
            entry['delta_'+field] = b-a if a is not None and b is not None else None
        comparison.append(entry)

    prior_tes = read(ROOT/'outputs/tes_pcc_scheduling_v1/summary.json')
    regression = []
    for case in cfg['scenarios']:
        prior = next((r for r in prior_tes['scenarios'] if r['id']==case['id']), None)
        if prior is None:
            continue
        current = solutions[(case['id'], 'tes_only')]
        passed = current['status']==prior['status']
        difference = None
        if current['feasible'] and prior['feasible']:
            difference = abs(current['metrics']['pcc_energy_mwh']-prior['optimized']['pcc_energy_mwh'])
            passed = passed and difference<1e-6
        regression.append({'case': case['id'], 'passed': passed, 'objective_difference_mwh': difference})
    v1_tes_unchanged = all(r['old_status']==r['new_status'] and
                           (r['delta_pcc_energy_mwh'] is None or abs(r['delta_pcc_energy_mwh'])<1e-6)
                           for r in comparison if r['strategy_id']=='tes_only')
    monotonic = all(not r['old_feasible'] or r['new_feasible'] for r in comparison)
    objective_nondegradation = all(r['delta_pcc_energy_mwh']<=1e-6 for r in comparison
                                  if r['old_status']=='optimal' and r['new_status']=='optimal')
    tests = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-m', 'unittest', 'discover',
                            '-s', 'tests', '-p', 'test_*tes*.py', '-v'], cwd=ROOT,
                           capture_output=True, text=True, encoding='utf-8')
    test_text = tests.stdout+tests.stderr
    count_match = re.search(r'Ran (\d+) tests?', test_text)
    after = {str(p.relative_to(ROOT)): sha(p) for p in sorted(protected)}
    fine_ok = all(r['held_10s']['minimum_chiller_headroom_mw_th']>=-1e-7 and
                  r['held_10s']['minimum_chiller_output_mw_th']>=-1e-7 and
                  r['held_10s']['reaggregation_error_mw']<1e-8 for r in summaries if r['feasible'])
    if len(records) != len(cfg['scenarios'])*len(cfg['strategies'])+3:
        raise AssertionError('Missing outputs')
    verified = all(all(r['checks'].values()) for r in records.values() if r['feasible'])
    resolved = all(r['status'] in ('optimal', 'infeasible', 'feasible_time_limit') for r in records.values())
    baseline_error = abs(old['no_storage']['energy_mwh']-float(p0.sum()*dt))
    endpoint_ok = all(abs(r['metrics']['ess_initial_energy_mwh']-2)<1e-7 and
                      abs(r['metrics']['ess_terminal_energy_mwh']-2)<1e-7
                      for name, r in records.items() if r['feasible'] and 'tes_only' not in name)
    verification = {'protected_file_count': len(before), 'protected_files_unchanged': before==after,
                    'protected_sha256': before, 'solution_file_count': len(records),
                    'solution_files_this_run': list(records), 'all_feasible_verified': verified,
                    'all_solver_statuses_resolved': resolved, 'tes_only_regression': regression,
                    'tes_only_regression_passed': all(r['passed'] for r in regression),
                    'v1_tes_only_unchanged': v1_tes_unchanged, 'previous_feasible_cases_remain_feasible': monotonic,
                    'optimal_objectives_do_not_worsen': objective_nondegradation,
                    'held_10s_chiller_and_reaggregation_passed': fine_ok,
                    'only_ess_energy_capacity_changed': changes==['energy_capacity_mwh'] and common_unchanged,
                    'changed_ess_parameters': changes, 'ess_endpoints_are_2mwh': endpoint_ok,
                    'pcc_caps_identical_to_v1': identical_caps, 'pcc_cap_max_differences_mw': cap_differences,
                    'no_storage_energy_difference_from_v1_mwh': baseline_error,
                    'unit_tests': {'passed': tests.returncode==0, 'count': int(count_match.group(1)) if count_match else None,
                                   'output': test_text}}
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(), 'duration_h': len(rows)*dt,
               'intervals': len(rows), 'no_storage': {'mean_pcc_mw': float(p0.mean()),
               'peak_pcc_mw': float(p0.max()), 'energy_mwh': float(p0.sum()*dt)},
               'coefficients': coeff, 'ess_config': cfg['ess'], 'results': summaries, 'sensitivity': [],
               'matched_hardware_comparison': matched, 'capacity_comparison': comparison,
               'comparison_baseline': cfg['comparison_baseline'],
               'scope': 'Only ESS capacity changed 2 to 4 MWh at fixed1MW AC; same5minute means, caps, losses, future knowledge and SOC endpoints.'}
    dump(OUT/'summary.json', summary)
    dump(OUT/'verification.json', verification)
    dump(OUT/'model_contract.json', {'experiment': cfg, 'inherited_tes_candidate': candidate,
         'scipy_version': scipy.__version__, 'common_conditions': 'Same load/horizon/caps; same50% endpoint SOC means1MWh in v1 versus2MWh in v2; own state restored in each.',
         'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in
             [ROOT/'src/aidc_facility/ess_tes_scheduler.py', CONFIG, Path(__file__)]}})
    np.savez_compressed(OUT/'schedules_5min.npz', **arrays)
    passed = (verified and resolved and fine_ok and before==after and
              verification['tes_only_regression_passed'] and v1_tes_unchanged and monotonic and
              identical_caps and objective_nondegradation and endpoint_ok and baseline_error<1e-8 and tests.returncode==0)
    if not passed:
        raise AssertionError('Validation failed; see verification.json')
    report_module = ROOT/'scripts/ess_tes_4mwh_report.py'
    if report_module.exists():
        module = importlib.import_module('ess_tes_4mwh_report')
        module.create_report(cfg, candidate, summary, verification, solutions, p0, hours, OUT)
        print('Report generated: '+str(OUT/'report_ko.html'), flush=True)
    else:
        print('Results complete; report module not yet present.', flush=True)
    print(json.dumps({'verified': passed, 'tests': verification['unit_tests']['count'],
                      'outputs': str(OUT), 'old_files_unchanged': before==after}, ensure_ascii=False), flush=True)


if __name__=='__main__':
    main()
