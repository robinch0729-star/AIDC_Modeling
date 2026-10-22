"""Conditional TES power-duration and recovery tests, not a DSO envelope.

Use frozen cooling demand and the existing energy operating kernel. The
analytical duration supports the adopted constant capacity-loss model only.
Every exported episode uses whole dispatch intervals and paid recovery.
"""
import math

from .tes_operating import energy_step, target_charge_request, validate_tes


def duration_limits(energy_mwh_th, discharge_mw_th, dt_h, tes_config):
    c = validate_tes(tes_config)
    energy, q, dt = map(float, (energy_mwh_th, discharge_mw_th, dt_h))
    if not all(math.isfinite(x) for x in (energy, q, dt)) or dt <= 0:
        raise ValueError("Require finite inputs and positive interval")
    low = c['energy_capacity_mwh_th'] * c['min_soc']
    high = c['energy_capacity_mwh_th'] * c['max_soc']
    if not low - 1e-10 <= energy <= high + 1e-10:
        raise ValueError("Energy outside SOC limits")
    if not 0 < q <= c['discharge_limit_mw_th']:
        raise ValueError("Discharge request must be positive and within rating")
    if c['hourly_retention_fraction'] != 1:
        raise ValueError("Analytical envelope requires capacity-only standby loss")
    loss = c['capacity_loss_fraction_per_hour'] * c['energy_capacity_mwh_th']
    duration = max(0., (energy - low) / (q / c['discharge_efficiency'] + loss))
    steps = int(math.floor(duration / dt + 1e-10))
    return {'continuous_duration_h': duration, 'full_steps': steps,
            'full_interval_duration_h': steps * dt,
            'unused_fractional_duration_h': max(0., duration - steps * dt)}


def pcc_from_thermal(base, charge, discharge, legacy, cop, pump_coefficient):
    """Same accounting boundary as the frozen operating candidate, no IT uplift."""
    if not math.isfinite(cop) or cop <= 0:
        raise ValueError("COP must be positive")
    if charge < 0 or discharge < 0 or charge * discharge > 1e-12:
        raise ValueError("Nonnegative, mutually exclusive thermal ports required")
    e, p = legacy['electrical'], legacy['plant']
    pump = pump_coefficient * (charge + discharge)
    thermal = base['chiller_mw_th'] + pump + charge - discharge
    chiller = thermal / cop
    condenser = p['condenser_pump_mw_per_mw_heat'] * (thermal + chiller)
    reject = thermal + chiller + condenser
    tower = p['tower_fan_mw_per_mw_heat'] * reject
    cooling = base['fan_mw'] + base['water_pump_mw'] + pump + chiller + condenser + tower
    downstream = base['it_mw'] / e['ups_efficiency'] + e['auxiliary_mw'] + cooling
    return {'pcc_mw': downstream / e['transformer_efficiency'],
            'chiller_mw_th': thermal, 'tes_pump_mw': pump, 'cooling_mw': cooling}


def marginal_pcc_coefficients(legacy, cop, pump_coefficient):
    p = legacy['plant']
    a, b = p['condenser_pump_mw_per_mw_heat'], p['tower_fan_mw_per_mw_heat']
    g = 1 / cop + (a + b * (1 + a)) * (1 + 1 / cop)
    eta = legacy['electrical']['transformer_efficiency']
    return {'reduction_mwe_per_mwth': ((1 - pump_coefficient) * g - pump_coefficient) / eta,
            'increase_mwe_per_mwth': ((1 + pump_coefficient) * g + pump_coefficient) / eta}


def evaluate_cycle(base_rows, start_index, initial_soc, discharge_rate, tes_config,
                   legacy, dt_h=1/12, discharge_steps=None, cop=5.):
    """Discharge full bins, then immediately recover to E0 on the same trace.

    Baseline waits over the identical horizon, compensates minimum-reserve loss
    if necessary, and restores E0 in the final bin through actual charging.
    Failure or trace truncation is returned, never repaired by wrapping data.
    """
    c = validate_tes(tes_config)
    if not isinstance(start_index, int) or not 0 <= start_index < len(base_rows):
        raise ValueError("Invalid start index")
    e0 = float(initial_soc) * c['energy_capacity_mwh_th']
    limits = duration_limits(e0, discharge_rate, dt_h, c)
    ndis = limits['full_steps'] if discharge_steps is None else discharge_steps
    if not isinstance(ndis, int) or ndis < 0:
        raise ValueError("discharge_steps must be a nonnegative integer")
    if ndis == 0:
        return {'status': 'no_full_discharge_interval', 'rows': [], 'limits': limits,
                'completed': False, 'service_feasible': False, 'start_index': start_index,
                'initial_soc': initial_soc, 'discharge_rate_mw_th': discharge_rate}
    low = c['min_soc'] * c['energy_capacity_mwh_th']
    loss = c['capacity_loss_fraction_per_hour'] * c['energy_capacity_mwh_th']
    capacity = legacy['plant']['chiller_capacity_mw_th']
    energy, rows, status = e0, [], 'trace_exhausted'
    for offset, source_index in enumerate(range(start_index, len(base_rows))):
        phase = 'discharge' if offset < ndis else 'recovery'
        if phase == 'recovery' and abs(energy - e0) < 1e-9:
            status = 'complete'
            break
        req = discharge_rate if phase == 'discharge' else target_charge_request(energy, e0, dt_h, c)
        maintenance = energy - loss * dt_h < low - 1e-10 and req >= 0
        if maintenance:
            req = target_charge_request(energy, low, dt_h, c)
        base = base_rows[source_index]
        step = energy_step(energy, req, dt_h, base['chiller_mw_th'], capacity, c)
        energy = step['energy_end_mwh_th']
        power = pcc_from_thermal(base, step['charge_mw_th'], step['discharge_mw_th'],
                                 legacy, cop, c['pump_mw_per_mw_th'])
        row = dict(step, **{'source_index': source_index, 'phase': phase,
                           'time_h': offset * dt_h, 'reserve_maintenance': maintenance,
                           'pcc_mw': power['pcc_mw'], 'base_chiller_mw_th': base['chiller_mw_th']})
        rows.append(row)
        if step['state_violation'] or step['physical_energy_violation']:
            status = 'storage_infeasible'
            break
        if phase == 'recovery' and abs(energy - e0) < 1e-9:
            status = 'complete'
            break
    if status != 'complete':
        return {'status': status, 'completed': False, 'rows': rows, 'limits': limits,
                'service_feasible': False, 'start_index': start_index,
                'initial_soc': initial_soc, 'discharge_rate_mw_th': discharge_rate}

    baseline_energy = e0
    for j, row in enumerate(rows):
        base = base_rows[row['source_index']]
        req = 0.
        if j == len(rows) - 1:
            req = target_charge_request(baseline_energy, e0, dt_h, c)
        elif baseline_energy - loss * dt_h < low - 1e-10:
            req = target_charge_request(baseline_energy, low, dt_h, c)
        st = energy_step(baseline_energy, req, dt_h, base['chiller_mw_th'], capacity, c)
        baseline_energy = st['energy_end_mwh_th']
        power = pcc_from_thermal(base, st['charge_mw_th'], st['discharge_mw_th'],
                                 legacy, cop, c['pump_mw_per_mw_th'])
        row.update({'baseline_pcc_mw': power['pcc_mw'], 'delta_pcc_mw': row['pcc_mw'] - power['pcc_mw']})
        row.update({'baseline_' + key: value for key, value in st.items()})
        if st['state_violation']:
            status = 'baseline_infeasible'
    if abs(baseline_energy - e0) > 1e-8:
        status = 'baseline_terminal_incomplete'

    dr = [r for r in rows if r['phase'] == 'discharge']
    rr = [r for r in rows if r['phase'] == 'recovery']
    shortfall = sum(max(0., discharge_rate - r['discharge_mw_th']) * dt_h for r in dr)
    full_prefix = 0
    for r in dr:
        if abs(r['discharge_mw_th'] - discharge_rate) > 1e-9:
            break
        full_prefix += 1
    residuals, continuity = [], []
    for prefix in ('', 'baseline_'):
        for j, r in enumerate(rows):
            residuals.append(abs(r[prefix+'energy_end_mwh_th'] - r[prefix+'energy_start_mwh_th'] -
                                 (c['charge_efficiency'] * r[prefix+'charge_mw_th'] -
                                  r[prefix+'discharge_mw_th'] / c['discharge_efficiency'] - loss) * dt_h))
            if j:
                continuity.append(abs(r[prefix+'energy_start_mwh_th'] - rows[j-1][prefix+'energy_end_mwh_th']))
    checks = {
        'same_initial_final_energy': abs(energy-e0) < 1e-8 and abs(baseline_energy-e0) < 1e-8,
        'full_discharge_delivered': shortfall < 1e-8,
        'energy_balance': max(residuals, default=0.) < 1e-9,
        'state_continuity': max(continuity, default=0.) < 1e-9,
        'soc_and_ratings': all(not r[p+'state_violation'] and not r[p+'physical_energy_violation']
                              and r[p+'charge_mw_th'] <= c['charge_limit_mw_th'] + 1e-9
                              and r[p+'discharge_mw_th'] <= c['discharge_limit_mw_th'] + 1e-9
                              and r[p+'charge_mw_th'] * r[p+'discharge_mw_th'] < 1e-12
                              for r in rows for p in ('', 'baseline_')),
        'chiller_capacity': all(-1e-9 <= r[p+'chiller_mw_th'] <= capacity + 1e-9
                                for r in rows for p in ('', 'baseline_'))}
    return {'status': status, 'completed': status == 'complete', 'rows': rows,
            'service_feasible': status == 'complete' and all(checks.values()),
            'start_index': start_index, 'initial_soc': initial_soc, 'discharge_rate_mw_th': discharge_rate,
            'limits': limits, 'requested_discharge_duration_h': len(dr)*dt_h,
            'full_discharge_duration_h': full_prefix*dt_h,
            'recovery_duration_h': len(rr)*dt_h, 'cycle_duration_h': len(rows)*dt_h,
            'energy_after_discharge_mwh_th': dr[-1]['energy_end_mwh_th'],
            'discharge_shortfall_mwh_th': shortfall,
            'discharge_energy_mwh_th': sum(r['discharge_mw_th']*dt_h for r in dr),
            'recovery_charge_mwh_th': sum(r['charge_mw_th']*dt_h for r in rr),
            'mean_pcc_reduction_mw': -sum(r['delta_pcc_mw'] for r in dr)/len(dr),
            'maximum_recovery_pcc_increase_mw': max((r['delta_pcc_mw'] for r in rr), default=0.),
            'discharge_electricity_reduction_mwh': -sum(r['delta_pcc_mw']*dt_h for r in dr),
            'recovery_electricity_increase_mwh': sum(r['delta_pcc_mw']*dt_h for r in rr),
            'net_electricity_increase_mwh': sum(r['delta_pcc_mw']*dt_h for r in rows),
            'global_peak_difference_over_episode_mw': max(r['pcc_mw'] for r in rows)-max(r['baseline_pcc_mw'] for r in rows),
            'energy_residual_max_mwh_th': max(residuals, default=0.), 'checks': checks}
