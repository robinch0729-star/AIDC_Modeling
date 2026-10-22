"""Review candidate: energy-conserving equipment-level PCC / cooling / TES.

PL13 eqs31,41,53-57: electrical/thermal balance and storage dispatch structure.
PL4 eqs8,10,12-16: equipment powers and heat paths. PL16 eqs8-9: lumped RC.
EnergyPlus mixed tank: C*dT/dt = Qdis-Qch+UA*(Tamb-T), HX flow limitations.
Integration, boundary choices, controller, equipment parameters and reduction
are our explicit scenario assumptions, not a verbatim reproduction of one paper.
Power MW; thermal power MW_th; stored energy MWh_th; time hours internally.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def validate(c):
    e,r,p,t = (c[k] for k in ('electrical','room','plant','tes'))
    for v in (e['ups_efficiency'],e['transformer_efficiency'],t['hx_effectiveness'],t['pump_efficiency']):
        if not 0 < v <= 1: raise ValueError('Efficiency must be in (0,1]')
    for v in (r['effective_capacity_mwh_per_k'],p['cop'],p['chiller_capacity_mw_th'],
              t['volume_m3'],t['water_density_kg_m3'],t['water_cp_j_kg_k'],
              t['maximum_flow_kg_s'],c['internal_step_s']):
        if not math.isfinite(v) or v <= 0: raise ValueError('Positive finite capacity required')
    if not t['source_chilled_c'] < t['minimum_c'] <= t['initial_c'] < t['maximum_discharge_supply_c'] < t['load_return_c']:
        raise ValueError('Inconsistent water temperature hierarchy')
    if not t['minimum_c'] <= t['charge_target_c'] < t['maximum_discharge_supply_c']:
        raise ValueError('Invalid charge target')
    if not r['min_c'] <= r['initial_c'] <= r['max_c']: raise ValueError('Invalid room initial state')
    if r['ua_mw_per_k'] < 0 or t['ua_mw_per_k'] < 0: raise ValueError('UA cannot be negative')
    if not 0 <= e['ups_heat_to_room_fraction'] <= 1: raise ValueError('Invalid heat fraction')
    if c['report_step_s']%c['internal_step_s']: raise ValueError('Report interval must be an integer multiple')
    slope=3*p['fan_variable_mw']*(r['coil_capacity_mw_th']/p['design_cooling_mw_th'])**2/p['design_cooling_mw_th']
    if slope>=1: raise ValueError('Fan model is outside monotonic heat-balance support')


def affine_thermal_step(temp, ambient, net_mw, capacity_mwh_k, ua_mw_k, dt_h):
    """Exact constant-input solution; q_env is the interval-mean ambient heat."""
    if ua_mw_k == 0:
        end = temp + net_mw*dt_h/capacity_mwh_k
    else:
        a = math.exp(-ua_mw_k*dt_h/capacity_mwh_k)
        end = ambient + (temp-ambient)*a + net_mw*(-math.expm1(-ua_mw_k*dt_h/capacity_mwh_k))/ua_mw_k
    q_env = capacity_mwh_k*(end-temp)/dt_h-net_mw
    return end, q_env


def tank_capacity(t):
    return t['volume_m3']*t['water_density_kg_m3']*t['water_cp_j_kg_k']/3.6e9


def maximum_discharge_tank_temperature(t):
    """Finite HX approach: load outlet is warmer than tank water, not equal."""
    return t['load_return_c']-(t['load_return_c']-t['maximum_discharge_supply_c'])/t['hx_effectiveness']


def tank_step(temp, requested_net, dt_h, t, qloop_without_tes, chiller_max):
    """Positive net = discharge; negative = charge. Never clip stored state.

    Feasible heat dispatch respects end-point supply temperature, source/load HX
    approach, mass flow, energy reserve and available chiller capacity. For a
    mixed tank the trajectory is monotone on each constant-input interval, so
    end-point worst temperatures also bound within-step HX capability.
    """
    cap=tank_capacity(t)
    def state(q):
        return affine_thermal_step(temp,t['ambient_c'],q,cap,t['ua_mw_per_k'],dt_h)
    def evaluate(magnitude, sign):
        q=sign*magnitude; end,leak=state(q)
        worst=max(temp,end) if sign>0 else min(temp,end)
        delta=(t['load_return_c']-worst) if sign>0 else (worst-t['source_chilled_c'])
        flow=magnitude/(t['hx_effectiveness']*t['water_cp_j_kg_k']/1e6*max(delta,1e-12))
        pump=flow*9.81*t['pump_head_m']/t['pump_efficiency']/1e6
        supply=t['load_return_c']-t['hx_effectiveness']*(t['load_return_c']-worst)
        if magnitude == 0: return True,end,leak,0.,0.,0.
        energy=cap*(t['reference_warm_c']-end)
        ok=flow <= t['maximum_flow_kg_s']+1e-10 and delta>0
        if sign>0:
            ok &= supply <= t['maximum_discharge_supply_c']+1e-10
            ok &= energy >= t['minimum_reserve_mwh_th']-1e-10
            ok &= magnitude <= qloop_without_tes+pump+1e-10
        else:
            ok &= end >= t['minimum_c']-1e-10
            ok &= qloop_without_tes+pump+magnitude <= chiller_max+1e-10
        return ok,end,leak,flow,pump,supply if sign>0 else 0.
    sign=1 if requested_net>=0 else -1
    rating=t['maximum_discharge_mw_th'] if sign>0 else t['maximum_charge_mw_th']
    hi=min(abs(requested_net),rating)
    if not evaluate(hi,sign)[0]:
        lo=0.
        for _ in range(36):
            mid=(lo+hi)/2
            if evaluate(mid,sign)[0]: lo=mid
            else: hi=mid
        hi=lo
    _,end,leak,flow,pump,supply=evaluate(hi,sign)
    return {'net':sign*hi,'end':end,'leak':leak,'flow':flow,'pump':pump,
            'curtailed':abs(requested_net)-hi,'supply':supply}


def equipment_aux(qcoil,p):
    ratio=max(qcoil,0.)/p['design_cooling_mw_th']
    return (p['fan_base_mw']+p['fan_variable_mw']*ratio**3,
            p['water_pump_base_mw']+p['water_pump_variable_mw']*ratio**3)


def monotonic_root(function, upper):
    if function(0.)>=0: return 0.
    if function(upper)<=0: return upper
    lo=0.; hi=upper
    for _ in range(30):
        mid=(lo+hi)/2
        if function(mid)>0: hi=mid
        else: lo=mid
    return (lo+hi)/2


def dispatch_command(time_h,total_h,temp,c):
    t,d=c['tes'],c['dispatch']; phase=time_h%24
    recovery=time_h>=total_h-d['terminal_recovery_hours']
    charge_window=d['charge_hours_elapsed_mod24'][0]<=phase<d['charge_hours_elapsed_mod24'][1]
    discharge_window=d['discharge_hours_elapsed_mod24'][0]<=phase<d['discharge_hours_elapsed_mod24'][1]
    if recovery or charge_window:
        # Exact heat needed to reach, or hold, the specified target this step.
        h=c['internal_step_s']/3600; cap=tank_capacity(t); ua=t['ua_mw_per_k']
        base,_=affine_thermal_step(temp,t['ambient_c'],0.,cap,ua,h)
        b=h/cap if ua==0 else -math.expm1(-ua*h/cap)/ua
        required=max((base-t['charge_target_c'])/b,0.)
        return -min(d['charge_command_mw_th'],required)
    return d['discharge_command_mw_th'] if discharge_window else 0.


def simulate(it_mw, c, with_tes=False, commands=None):
    """IT samples are interval-mean inputs at internal_step_s, never rescaled."""
    validate(c); it=np.asarray(it_mw,float)
    if it.ndim!=1 or len(it)==0 or not np.isfinite(it).all() or np.any(it<0):
        raise ValueError('Nonnegative finite IT time series required')
    if commands is not None and len(commands)!=len(it): raise ValueError('Command length mismatch')
    e,r,p,t=(c[k] for k in ('electrical','room','plant','tes'))
    h=c['internal_step_s']/3600; room=r['initial_c']; tank=t['initial_c']; cap=tank_capacity(t)
    rows=[]
    for k,pit in enumerate(it):
        ups=pit*(1/e['ups_efficiency']-1)
        heat0=pit+ups*e['ups_heat_to_room_fraction']+e['auxiliary_mw']
        feedback=r['effective_capacity_mwh_per_k']/(r['tracking_time_s']/3600)*(room-r['setpoint_c'])
        desired=monotonic_root(lambda q:q-heat0-equipment_aux(q,p)[0]-r['ua_mw_per_k']*(r['ambient_c']-room)-feedback,
                               r['coil_capacity_mw_th'])
        _,water_guess=equipment_aux(desired,p)
        requested=float(commands[k]) if commands is not None else dispatch_command(k*h,len(it)*h,tank,c)
        if not with_tes: requested=0.
        ts=tank_step(tank,requested,h,t,desired+water_guess,p['chiller_capacity_mw_th']) if with_tes else {
            'net':0.,'end':tank,'leak':0.,'flow':0.,'pump':0.,'curtailed':0.,'supply':0.}
        net=ts['net']
        available=p['chiller_capacity_mw_th']+net-ts['pump']
        qcoil=min(desired,monotonic_root(lambda q:q+equipment_aux(q,p)[1]-available,r['coil_capacity_mw_th']))
        fan,water=equipment_aux(qcoil,p)
        qch=qcoil+water+ts['pump']-net
        if qch < -1e-7: raise AssertionError('Unmodeled excess cold production')
        pch=qch/p['cop']
        condenser=p['condenser_pump_mw_per_mw_heat']*(qch+pch)
        reject=qch+pch+condenser
        tower=p['tower_fan_mw_per_mw_heat']*reject
        cooling=fan+water+ts['pump']+pch+condenser+tower
        downstream=pit+ups+e['auxiliary_mw']+cooling
        transformer=downstream*(1/e['transformer_efficiency']-1)
        pcc=downstream+transformer
        end_room,env=affine_thermal_step(room,r['ambient_c'],heat0+fan-qcoil,
                                       r['effective_capacity_mwh_per_k'],r['ua_mw_per_k'],h)
        energy0=cap*(t['reference_warm_c']-tank); energy1=cap*(t['reference_warm_c']-ts['end'])
        rows.append({
            'time_s':k*c['internal_step_s'],'it_mw':pit,'pcc_mw':pcc,'cooling_mw':cooling,
            'ups_loss_mw':ups,'transformer_loss_mw':transformer,'auxiliary_mw':e['auxiliary_mw'],
            'fan_mw':fan,'water_pump_mw':water,'tes_pump_mw':ts['pump'],'chiller_mw':pch,
            'condenser_pump_mw':condenser,'tower_fan_mw':tower,'heat_rejection_mw_th':reject,
            'coil_mw_th':qcoil,'chiller_mw_th':qch,'requested_coil_mw_th':desired,
            'thermal_control_shortfall_mw_th':max(desired-qcoil,0.),
            'room_start_c':room,'room_end_c':end_room,'room_ambient_heat_mw_th':env,
            'tank_start_c':tank,'tank_end_c':ts['end'],'tes_energy_start_mwh_th':energy0,
            'tes_energy_end_mwh_th':energy1,'tes_command_mw_th':requested,'tes_net_mw_th':net,
            'tes_charge_mw_th':max(-net,0.),'tes_discharge_mw_th':max(net,0.),
            'tes_flow_kg_s':ts['flow'],'tes_heat_leak_mw_th':ts['leak'],
            'tes_discharge_supply_c':ts['supply'],
            'tes_command_curtailed_mw_th':ts['curtailed'],
            'electrical_balance_mw':pcc-(pit+ups+e['auxiliary_mw']+cooling+transformer),
            'water_balance_mw_th':qch+net-qcoil-water-ts['pump'],
            'room_balance_mwh_th':r['effective_capacity_mwh_per_k']*(end_room-room)-(heat0+fan+env-qcoil)*h,
            'tank_balance_mwh_th':(energy1-energy0)-(-net-ts['leak'])*h if with_tes else 0.
        })
        room=end_room; tank=ts['end']
    return pd.DataFrame(rows)


def summarize(frame,c,with_tes):
    h=c['internal_step_s']/3600; r=c['room']; t=c['tes']; f=frame
    n=c['report_step_s']//c['internal_step_s']
    if len(f)%n: raise ValueError('Complete output bins required')
    p5=f.pcc_mw.to_numpy().reshape(-1,n).mean(axis=1)
    dis=f.tes_discharge_mw_th>1e-6
    limits={
        'room_temperature_violations':int(((f.room_end_c<r['min_c']-1e-8)|(f.room_end_c>r['max_c']+1e-8)).sum()),
        'discharge_supply_violations':int((dis & (f.tes_discharge_supply_c>t['maximum_discharge_supply_c']+1e-7)).sum()),
        'tank_minimum_violations':int((f.tank_end_c<t['minimum_c']-1e-7).sum()) if with_tes else 0,
        'flow_violations':int((f.tes_flow_kg_s>t['maximum_flow_kg_s']+1e-6).sum()),
        'chiller_capacity_violations':int(((f.chiller_mw_th>c['plant']['chiller_capacity_mw_th']+1e-6)|(f.chiller_mw_th< -1e-6)).sum()),
        'simultaneous_charge_discharge':int(((f.tes_charge_mw_th>1e-6)&(f.tes_discharge_mw_th>1e-6)).sum()),
        'reserve_violations':int((f.tes_energy_end_mwh_th<t['minimum_reserve_mwh_th']-1e-7).sum()) if with_tes else 0
    }
    return {
        'with_tes':with_tes,'duration_h':len(f)*h,'it_mean_mw':float(f.it_mw.mean()),
        'pcc_mean_mw':float(f.pcc_mw.mean()),'pcc_5min_peak_mw':float(p5.max()),
        'pcc_energy_mwh':float(f.pcc_mw.sum()*h),'cooling_energy_mwh':float(f.cooling_mw.sum()*h),
        'pue_energy_ratio':float(f.pcc_mw.sum()/f.it_mw.sum()),
        'room_min_c':float(min(r['initial_c'],f.room_end_c.min())),
        'room_max_c':float(max(r['initial_c'],f.room_end_c.max())),
        'tank_min_c':float(min(t['initial_c'],f.tank_end_c.min())),
        'tank_max_c':float(max(t['initial_c'],f.tank_end_c.max())),
        'tank_final_c':float(f.tank_end_c.iloc[-1]),
        'tes_energy_change_mwh_th':float(f.tes_energy_end_mwh_th.iloc[-1]-f.tes_energy_start_mwh_th.iloc[0]),
        'tes_charge_mwh_th':float(f.tes_charge_mw_th.sum()*h),
        'tes_discharge_mwh_th':float(f.tes_discharge_mw_th.sum()*h),
        'tes_heat_leak_mwh_th':float(f.tes_heat_leak_mw_th.sum()*h),
        'tes_command_curtailed_mwh_th':float(f.tes_command_curtailed_mw_th.sum()*h),
        'thermal_control_shortfall_mwh_th':float(f.thermal_control_shortfall_mw_th.sum()*h),
        'max_abs_balances':{k:float(f[k].abs().max()) for k in f if '_balance_' in k},
        'constraints':limits,'feasible_within_declared_checks':all(v==0 for v in limits.values()),
        'mean_components_mw':{k:float(f[k].mean()) for k in ['it_mw','chiller_mw','fan_mw','water_pump_mw',
            'tes_pump_mw','condenser_pump_mw','tower_fan_mw','ups_loss_mw','transformer_loss_mw','auxiliary_mw']}
    }


def aggregate(frame,c):
    n=c['report_step_s']//c['internal_step_s']
    if n*c['internal_step_s']!=c['report_step_s'] or len(frame)%n: raise ValueError('Incompatible intervals')
    groups=np.arange(len(frame))//n
    result=frame.groupby(groups).mean()
    result['time_s']=np.arange(len(result))*c['report_step_s']
    for col in frame:
        if '_start_' in col: result[col]=frame.groupby(groups)[col].first()
        if '_end_' in col: result[col]=frame.groupby(groups)[col].last()
    # Temperature extrema are needed: average temperature alone cannot certify feasibility.
    result['room_min_c']=frame.groupby(groups).room_end_c.min()
    result['room_max_c']=frame.groupby(groups).room_end_c.max()
    result['tank_min_c']=frame.groupby(groups).tank_end_c.min()
    result['tank_max_c']=frame.groupby(groups).tank_end_c.max()
    return result
