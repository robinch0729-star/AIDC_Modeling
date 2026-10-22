"""Independent saved-array audit; does not call the simulator or its checks."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/pcc_cooling_tes_candidate_v1'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def maxabs(x): return float(np.max(np.abs(x)))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    c=read(OUT/'generation_config.json'); columns=read(OUT/'internal_columns.json')
    saved=np.load(OUT/'internal_profiles.npz'); csv=pd.read_csv(OUT/'pcc_profiles_5min.csv')
    h=c['internal_step_s']/3600; r=c['room']; t=c['tes']; p=c['plant']; e=c['electrical']
    cap=t['volume_m3']*t['water_density_kg_m3']*t['water_cp_j_kg_k']/3.6e9
    checks={}
    for name in saved.files:
        f=pd.DataFrame(saved[name],columns=columns); active=f.tes_discharge_mw_th>1e-6
        components=f[['it_mw','ups_loss_mw','transformer_loss_mw','auxiliary_mw','fan_mw',
            'water_pump_mw','tes_pump_mw','chiller_mw','condenser_pump_mw','tower_fan_mw']].sum(axis=1)
        heat=f.it_mw+f.ups_loss_mw*e['ups_heat_to_room_fraction']+f.auxiliary_mw+f.fan_mw
        room_balance=r['effective_capacity_mwh_per_k']*(f.room_end_c-f.room_start_c)-(heat+f.room_ambient_heat_mw_th-f.coil_mw_th)*h
        tank_balance=(f.tes_energy_end_mwh_th-f.tes_energy_start_mwh_th)-(f.tes_charge_mw_th-f.tes_discharge_mw_th-f.tes_heat_leak_mw_th)*h
        water_balance=f.chiller_mw_th+f.tes_discharge_mw_th-f.tes_charge_mw_th-f.coil_mw_th-f.water_pump_mw-f.tes_pump_mw
        ar=np.exp(-r['ua_mw_per_k']*h/r['effective_capacity_mwh_per_k'])
        room_exact=r['ambient_c']+(f.room_start_c-r['ambient_c'])*ar+(heat-f.coil_mw_th)*(1-ar)/r['ua_mw_per_k']
        if name!='no_tes':
            at=np.exp(-t['ua_mw_per_k']*h/cap)
            tank_exact=t['ambient_c']+(f.tank_start_c-t['ambient_c'])*at+f.tes_net_mw_th*(1-at)/t['ua_mw_per_k']
            tank_error=maxabs(f.tank_end_c-tank_exact)
        else: tank_error=maxabs(f.tank_end_c-t['initial_c'])
        supply=t['load_return_c']-t['hx_effectiveness']*(t['load_return_c']-f[['tank_start_c','tank_end_c']].max(axis=1))
        rated_errors=max(float((f.tes_charge_mw_th-t['maximum_charge_mw_th']).max()),
                         float((f.tes_discharge_mw_th-t['maximum_discharge_mw_th']).max()),0.)
        b=csv[csv.scenario.eq(name)]; n=c['report_step_s']//c['internal_step_s']
        checks[name]={
            'pcc_component_sum_error_mw':maxabs(f.pcc_mw-components),
            'chiller_cop_error_mw':maxabs(f.chiller_mw-f.chiller_mw_th/p['cop']),
            'room_energy_error_mwh_th':maxabs(room_balance),'tank_energy_error_mwh_th':maxabs(tank_balance),
            'water_balance_error_mw_th':maxabs(water_balance),
            'room_exact_solution_error_c':maxabs(f.room_end_c-room_exact),
            'tank_exact_solution_error_c':tank_error,
            'tank_temperature_to_energy_error_mwh_th':maxabs(f.tes_energy_end_mwh_th-cap*(t['reference_warm_c']-f.tank_end_c)),
            'five_minute_power_reaggregation_error_mw':maxabs(b.pcc_mw.to_numpy()-f.pcc_mw.to_numpy().reshape(-1,n).mean(axis=1)),
            'pcc_energy_reaggregation_error_mwh':abs(float(b.pcc_mw.sum()*300/3600-f.pcc_mw.sum()*h)),
            'supply_temperature_violations':int((supply[active]>t['maximum_discharge_supply_c']+1e-7).sum()),
            'flow_violations':int((f.tes_flow_kg_s>t['maximum_flow_kg_s']+1e-6).sum()),
            'rate_limit_excess_mw_th':rated_errors}
        for key,value in checks[name].items():
            if 'error' in key: assert value<1e-7,(name,key,value)
            elif 'violations' in key: assert value==0,(name,key,value)
        assert rated_errors<1e-7
    manifest=read(OUT/'source_manifest.json')
    assert all(digest(ROOT/f)==v for f,v in manifest.items())
    report=(OUT/'report_ko.html').read_text(encoding='utf-8')
    links=re.findall(r'(?:href|src)="([^"#]+)"',report)
    local=[x for x in links if not x.startswith(('https:','http:'))]
    assert all((OUT/x).is_file() for x in local)
    for section in ('*1','*2','*3','*4','*5','*6'): assert section in report
    result={'all_checks_passed':True,'saved_array_checks':checks,
            'protected_it_files_unchanged':len(manifest),'local_report_links_valid':len(local),
            'scope':'Algebra, integration and constraints under stated assumptions; not site validation'}
    (OUT/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    artifacts=[ROOT/'configs/pcc_cooling_tes_candidate_v1.json',ROOT/'src/aidc_facility/pcc_cooling_tes.py',
               ROOT/'scripts/run_pcc_cooling_tes_candidate.py',Path(__file__),ROOT/'tests/test_pcc_cooling_tes.py']
    artifacts+=[x for x in OUT.iterdir() if x.is_file() and x.name!='artifact_manifest.json']
    (OUT/'artifact_manifest.json').write_text(json.dumps({str(x.relative_to(ROOT)):digest(x) for x in artifacts},indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
