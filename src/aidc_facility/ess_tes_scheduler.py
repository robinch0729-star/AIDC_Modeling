"""Joint ESS--TES operating benchmark with hard exogenous PCC limits.

The ESS charge/discharge ports are AC powers referred to the PCC. Efficiencies
are end-to-end between those ports and stored energy: no additional transformer
or UPS efficiency is applied to these powers. No extra battery cooling or
converter model is implied. TES retains the existing frozen-service thermal
accounting. A disabled device is absent, not installed and idling: its ports,
states and losses are zero and its requested initial/terminal SOC is inactive.

The objective is total imported energy, not tariff cost or battery degradation.
The model has perfect foresight; it is not a controller or a grid-derived DOE.
All feasible schedules require an independent energy/plant replay. No slack,
state clipping, post-hoc recovery, or hidden curtailment is accepted.
"""

from collections.abc import Mapping
import math
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .tes_flexibility import marginal_pcc_coefficients, pcc_from_thermal
from .tes_operating import energy_step
from .tes_scheduler import _number, _validated_inputs


def validate_ess(config):
    """Return a numeric copy of a PCC-referred ESS operating parameter set."""
    if not isinstance(config, Mapping):
        raise ValueError('ess_config must be a mapping')
    keys = ('energy_capacity_mwh', 'charge_limit_mw', 'discharge_limit_mw',
            'charge_efficiency', 'discharge_efficiency', 'min_soc', 'max_soc')
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError('Missing ESS parameters: ' + ', '.join(missing))
    c = {key: _number('ess.' + key, config[key]) for key in keys}
    c['hourly_retention_fraction'] = _number(
        'ess.hourly_retention_fraction', config.get('hourly_retention_fraction', 1.))
    if c['energy_capacity_mwh'] <= 0:
        raise ValueError('ESS energy_capacity_mwh must be positive')
    if not 0 <= c['min_soc'] < c['max_soc'] <= 1:
        raise ValueError('ESS requires 0 <= min_soc < max_soc <= 1')
    if min(c['charge_limit_mw'], c['discharge_limit_mw']) < 0:
        raise ValueError('ESS charge and discharge limits must be nonnegative')
    if not all(0 < c[key] <= 1 for key in
               ('charge_efficiency', 'discharge_efficiency', 'hourly_retention_fraction')):
        raise ValueError('ESS efficiencies and retention must be in (0, 1]')
    return c


def _finite_optional(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def solve_joint_schedule(base_rows, pcc_upper_mw, tes_config, legacy, ess_config,
                         dt_h=1/12, initial_tes_soc=.5, terminal_tes_soc=.5,
                         initial_ess_soc=.5, terminal_ess_soc=.5, cop=5.,
                         use_tes=True, use_ess=True, objective='energy',
                         time_limit_s=45):
    """Minimize import energy with enabled ESS/TES states and terminal recovery.

    Inputs ``base_rows`` and TES/legacy dictionaries use the TES scheduler's
    contract. Caps may be a scalar or one value per bin. The hard limit applies
    to the supplied bin means, not to intra-bin power. ESS retention is hourly
    stored-energy retention; exact constant-command integration avoids changing
    physical self-discharge when the timestep is changed.

    ``schedule`` is populated only for independently verified solver incumbents.
    ``optimal`` additionally requires solver status 0. Time-limit incumbents are
    labelled ``feasible_time_limit`` when verified, never labelled infeasible.
    Every returned value is JSON serializable. Disabled energy-state arrays are
    zero placeholders, not physical SOC trajectories.
    """
    if objective != 'energy':
        raise ValueError("Only objective='energy' is implemented")
    if not isinstance(use_tes, (bool, np.bool_)) or not isinstance(use_ess, (bool, np.bool_)):
        raise ValueError('use_tes and use_ess must be boolean')
    use_tes, use_ess = bool(use_tes), bool(use_ess)
    rows, caps, tc, legacy, dt, ts0, tsn, cop, time_limit = _validated_inputs(
        base_rows, pcc_upper_mw, tes_config, legacy, dt_h,
        initial_tes_soc, terminal_tes_soc, cop, time_limit_s)
    ec = validate_ess(ess_config)
    es0, esn = (_number('initial_ess_soc', initial_ess_soc),
                _number('terminal_ess_soc', terminal_ess_soc))
    if not (ec['min_soc'] <= es0 <= ec['max_soc'] and ec['min_soc'] <= esn <= ec['max_soc']):
        raise ValueError('Initial and terminal ESS SOC must lie inside ESS SOC bounds')
    n = len(rows)
    tcap, ecap = tc['energy_capacity_mwh_th'], ec['energy_capacity_mwh']
    tmin, tmax = (tc['min_soc'] * tcap, tc['max_soc'] * tcap) if use_tes else (0., 0.)
    emin, emax = (ec['min_soc'] * ecap, ec['max_soc'] * ecap) if use_ess else (0., 0.)
    te0, ten = (ts0 * tcap, tsn * tcap) if use_tes else (0., 0.)
    ee0, een = (es0 * ecap, esn * ecap) if use_ess else (0., 0.)
    qcmax, qdmax = (tc['charge_limit_mw_th'], tc['discharge_limit_mw_th']) if use_tes else (0., 0.)
    pcmax, pdmax = (ec['charge_limit_mw'], ec['discharge_limit_mw']) if use_ess else (0., 0.)
    tloss = tc['capacity_loss_fraction_per_hour'] * tcap if use_tes else 0.
    pump = tc['pump_mw_per_mw_th']
    chill_capacity = legacy['plant']['chiller_capacity_mw_th']
    decay = -math.log(ec['hourly_retention_fraction']) if use_ess else 0.
    eret = math.exp(-decay * dt)
    beta = -math.expm1(-decay * dt) / decay if decay else dt
    p0 = np.array([pcc_from_thermal(row, 0., 0., legacy, cop, pump)['pcc_mw'] for row in rows])
    marginal = marginal_pcc_coefficients(legacy, cop, pump)
    a, b = marginal['increase_mwe_per_mwth'], marginal['reduction_mwe_per_mwth']

    # Each device: charge[n], discharge[n], energy[n+1], binary mode[n].
    tqc, tqd, te, tz = 0, n, 2*n, 3*n+1
    epc, epd, ee, ez = 4*n+1, 5*n+1, 6*n+1, 7*n+2
    nv = 8*n+2
    lower = np.zeros(nv)
    upper = np.zeros(nv)
    for start, count, low, high in (
        (tqc, n, 0., qcmax), (tqd, n, 0., qdmax), (te, n+1, tmin, tmax),
        (tz, n, 0., float(use_tes)), (epc, n, 0., pcmax),
        (epd, n, 0., pdmax), (ee, n+1, emin, emax), (ez, n, 0., float(use_ess))):
        lower[start:start+count], upper[start:start+count] = low, high
    for index, val in ((te, te0), (te+n, ten), (ee, ee0), (ee+n, een)):
        lower[index] = upper[index] = val
    integrality = np.zeros(nv, dtype=np.uint8)
    integrality[tz:tz+n] = integrality[ez:ez+n] = 1
    objective_coeff = np.zeros(nv)
    objective_coeff[tqc:tqc+n], objective_coeff[tqd:tqd+n] = a*dt, -b*dt
    objective_coeff[epc:epc+n], objective_coeff[epd:epd+n] = dt, -dt
    ri, ci, values, lb, ub = [], [], [], [], []

    def constraint(items, low, high):
        index = len(lb)
        for col, val in items:
            ri.append(index)
            ci.append(col)
            values.append(val)
        lb.append(low)
        ub.append(high)

    for k, base in enumerate(rows):
        constraint(((te+k+1, 1.), (te+k, -1.),
                    (tqc+k, -dt*tc['charge_efficiency']),
                    (tqd+k, dt/tc['discharge_efficiency'])), -dt*tloss, -dt*tloss)
        constraint(((ee+k+1, 1.), (ee+k, -eret),
                    (epc+k, -beta*ec['charge_efficiency']),
                    (epd+k, beta/ec['discharge_efficiency'])), 0., 0.)
        constraint(((tqc+k, 1.), (tz+k, -qcmax)), -np.inf, 0.)
        constraint(((tqd+k, 1.), (tz+k, qdmax)), -np.inf, qdmax)
        constraint(((epc+k, 1.), (ez+k, -pcmax)), -np.inf, 0.)
        constraint(((epd+k, 1.), (ez+k, pdmax)), -np.inf, pdmax)
        constraint(((tqc+k, 1+pump), (tqd+k, -(1-pump))),
                   -base['chiller_mw_th'], chill_capacity-base['chiller_mw_th'])
        constraint(((tqc+k, a), (tqd+k, -b), (epc+k, 1.), (epd+k, -1.)),
                   -p0[k], caps[k]-p0[k])
    matrix = coo_matrix((values, (ri, ci)), shape=(len(lb), nv)).tocsc()
    # HiGHS defaults to 1e-6 MIP integrality tolerance, looser than this
    # kernel's 1e-7 replay tolerance. Request 1e-8 so a solver-accepted mode
    # does not fail our stricter audit. SciPy 1.17.1 _milp_iv deliberately
    # forwards non-listed options to HiGHS. Suppress only its notification
    # for this verified option; do not relax any independent physical check.
    # HiGHS documents this as its MIP integrality tolerance:
    # https://ergo-code.github.io/HiGHS/stable/options/definitions/#mip_feasibility_tolerance
    mip_integrality_tolerance = 1e-8
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', category=RuntimeWarning,
            message=r"Unrecognized options detected: \{'mip_feasibility_tolerance'\}.*")
        result = milp(objective_coeff, integrality=integrality, bounds=Bounds(lower, upper),
                      constraints=LinearConstraint(matrix, np.array(lb), np.array(ub)),
                      options={'time_limit': time_limit, 'mip_rel_gap': 1e-8, 'presolve': True,
                               'mip_feasibility_tolerance': mip_integrality_tolerance})
    status_names = {0: 'optimal', 1: 'time_limit', 2: 'infeasible', 3: 'unbounded', 4: 'solver_error'}
    output = {
        'status': status_names.get(result.status, 'solver_error'), 'success': False,
        'feasible': False, 'optimal': False, 'objective': objective,
        'solver': {'status_code': int(result.status), 'message': str(result.message),
                   'success': bool(result.success),
                   'mip_gap': _finite_optional(getattr(result, 'mip_gap', None)),
                   'mip_node_count': _finite_optional(getattr(result, 'mip_node_count', None)),
                   'objective_increment_mwh': _finite_optional(getattr(result, 'fun', None)),
                   'dual_bound_increment_mwh': _finite_optional(getattr(result, 'mip_dual_bound', None)),
                   'requested_mip_integrality_tolerance': mip_integrality_tolerance,
                   'time_limit_s': time_limit},
        'rows': [], 'schedule': None, 'checks': {}, 'metrics': None,
        'contract': {
            'limit_source': 'exogenous_input_not_network_derived', 'perfect_information': True,
            'hard_pcc_limits': True, 'slack_variables': False, 'dt_h': dt, 'intervals': n,
            'use_tes': use_tes, 'use_ess': use_ess, 'disabled_device': 'absent_zero_states_and_losses',
            'initial_tes_soc': ts0 if use_tes else None, 'terminal_tes_soc': tsn if use_tes else None,
            'initial_ess_soc': es0 if use_ess else None, 'terminal_ess_soc': esn if use_ess else None,
            'cop': cop, 'capacity_standby_loss_mw_th': tloss,
            'pcc_increase_mwe_per_mwth': a, 'pcc_reduction_mwe_per_mwth': b,
            'ess_boundary': 'PCC_referred_AC_end_to_end_efficiencies_no_extra_transformer_factor',
            'ess_retention_model': 'exact_continuous_proportional_self_discharge',
            'ess_hourly_retention_fraction': ec['hourly_retention_fraction'],
            'ess_bin_retention': eret, 'ess_input_integration_h': beta,
            'battery_degradation_cost': False, 'tariff_cost': False}}
    if result.status not in (0, 1) or result.x is None:
        return output
    x = np.asarray(result.x, dtype=float)
    if x.shape != (nv,) or not np.all(np.isfinite(x)):
        output['status'] = 'verification_failed'
        output['verification_error'] = 'Nonfinite or wrong-size solver incumbent'
        return output
    tol = 1e-7
    qc, qd = x[tqc:tqc+n].copy(), x[tqd:tqd+n].copy()
    pc, pd = x[epc:epc+n].copy(), x[epd:epd+n].copy()
    for ports in (qc, qd, pc, pd):
        ports[np.abs(ports) < 1e-10] = 0.  # numerical signed zero only
    replay_rows, tstates, estates = [], [te0], [ee0]
    tstate, estate = te0, ee0
    try:
        for k, base in enumerate(rows):
            if min(qc[k], qd[k], pc[k], pd[k]) < 0 or qc[k]*qd[k] > 1e-12 or pc[k]*pd[k] > 1e-12:
                raise ValueError('Negative or simultaneous charge/discharge in incumbent')
            tstart, estart = tstate, estate
            if use_tes:
                thermal = energy_step(tstate, float(qd[k]-qc[k]), dt,
                                      base['chiller_mw_th'], chill_capacity, tc)
                tstate = thermal['energy_end_mwh_th']
            else:
                thermal = {'energy_end_mwh_th': 0., 'curtailed_mw_th': 0.,
                           'energy_balance_error_mwh_th': 0., 'standby_loss_mwh_th': 0.,
                           'charge_conversion_loss_mwh_th': 0., 'discharge_conversion_loss_mwh_th': 0.}
            power = pcc_from_thermal(base, float(qc[k]), float(qd[k]), legacy, cop, pump)
            # Independent, unclipped exact ESS state integration and loss accounting.
            erate = ec['charge_efficiency']*float(pc[k])-float(pd[k])/ec['discharge_efficiency']
            if use_ess:
                estate = (math.exp(-decay*dt)*estart +
                          (-math.expm1(-decay*dt)/decay if decay else dt)*erate)
            else:
                estate = 0.
            ess_standby = estart + erate*dt - estate if decay else 0.
            ess_charge_loss = (1-ec['charge_efficiency'])*float(pc[k])*dt
            ess_discharge_loss = (1/ec['discharge_efficiency']-1)*float(pd[k])*dt
            ess_balance = estate-estart-((float(pc[k])-float(pd[k]))*dt-
                                        ess_charge_loss-ess_discharge_loss-ess_standby)
            pcc = power['pcc_mw']+float(pc[k])-float(pd[k])
            tstates.append(tstate)
            estates.append(estate)
            replay_rows.append({
                'source_index': k, 'time_h': k*dt, 'pcc_mw': pcc,
                'pcc_upper_mw': float(caps[k]), 'no_storage_pcc_mw': float(p0[k]),
                'thermal_pcc_mw': power['pcc_mw'], 'delta_pcc_mw': pcc-float(p0[k]),
                'base_chiller_mw_th': base['chiller_mw_th'], 'chiller_mw_th': power['chiller_mw_th'],
                'tes_charge_mw_th': float(qc[k]), 'tes_discharge_mw_th': float(qd[k]),
                'tes_energy_start_mwh_th': tstart, 'tes_energy_end_mwh_th': tstate,
                'tes_soc_start': tstart/tcap if use_tes else None,
                'tes_soc_end': tstate/tcap if use_tes else None,
                'tes_pump_mw': power['tes_pump_mw'],
                'tes_standby_loss_mwh_th': thermal['standby_loss_mwh_th'],
                'tes_charge_conversion_loss_mwh_th': thermal['charge_conversion_loss_mwh_th'],
                'tes_discharge_conversion_loss_mwh_th': thermal['discharge_conversion_loss_mwh_th'],
                'tes_energy_balance_error_mwh_th': thermal['energy_balance_error_mwh_th'],
                'tes_curtailed_mw_th': thermal['curtailed_mw_th'],
                'ess_charge_mw': float(pc[k]), 'ess_discharge_mw': float(pd[k]),
                'ess_energy_start_mwh': estart, 'ess_energy_end_mwh': estate,
                'ess_soc_start': estart/ecap if use_ess else None,
                'ess_soc_end': estate/ecap if use_ess else None,
                'ess_standby_loss_mwh': ess_standby,
                'ess_charge_conversion_loss_mwh': ess_charge_loss,
                'ess_discharge_conversion_loss_mwh': ess_discharge_loss,
                'ess_energy_balance_error_mwh': ess_balance})
    except ValueError as exc:
        output['status'] = 'verification_failed'
        output['verification_error'] = str(exc)
        return output
    tstates, estates = np.array(tstates), np.array(estates)
    power = np.array([row['pcc_mw'] for row in replay_rows])
    linear_value = matrix @ x
    linerr = max(float(np.max(np.maximum(np.array(lb)-linear_value, 0.))),
                 float(np.max(np.maximum(linear_value-np.array(ub), 0.))))
    bounderr = max(float(np.max(np.maximum(lower-x, 0.))),
                   float(np.max(np.maximum(x-upper, 0.))))
    binerr = max(float(np.max(np.abs(x[start:start+n]-np.rint(x[start:start+n])))) for start in (tz, ez))
    treplay = float(np.max(np.abs(tstates-x[te:te+n+1])))
    ereplay = float(np.max(np.abs(estates-x[ee:ee+n+1])))
    terr, eerr = abs(float(tstates[-1])-ten), abs(float(estates[-1])-een)
    caperr = float(np.max(np.maximum(power-caps, 0.)))
    curtail = max(row['tes_curtailed_mw_th'] for row in replay_rows)
    tbalance = max(abs(row['tes_energy_balance_error_mwh_th']) for row in replay_rows)
    ebalance = max(abs(row['ess_energy_balance_error_mwh']) for row in replay_rows)
    checks = {
        'linear_constraints': linerr <= tol, 'variable_bounds': bounderr <= tol,
        'binary_modes': binerr <= tol,
        'tes_mutually_exclusive_ports': bool(np.all(qc*qd <= 1e-12)),
        'ess_mutually_exclusive_ports': bool(np.all(pc*pd <= 1e-12)),
        'pcc_upper_limit': caperr <= tol, 'pcc_nonnegative': bool(np.min(power) >= -tol),
        'tes_storage_bounds': bool(np.min(tstates) >= tmin-tol and np.max(tstates) <= tmax+tol),
        'ess_storage_bounds': bool(np.min(estates) >= emin-tol and np.max(estates) <= emax+tol),
        'tes_terminal_energy': terr <= tol, 'ess_terminal_energy': eerr <= tol,
        'chiller_capacity': all(-tol <= row['chiller_mw_th'] <= chill_capacity+tol for row in replay_rows),
        'no_tes_dispatch_curtailment': curtail <= tol,
        'independent_tes_energy_replay': treplay <= tol,
        'independent_ess_energy_replay': ereplay <= tol,
        'tes_energy_balance': tbalance <= tol, 'ess_energy_balance': ebalance <= tol,
        'ess_standby_nonnegative': all(row['ess_standby_loss_mwh'] >= -tol for row in replay_rows),
        'pcc_linear_equivalence': bool(np.max(np.abs(power-(p0+a*qc-b*qd+pc-pd))) <= tol),
        'disabled_devices_absent': bool((use_tes or
                                        (np.all(qc == 0.) and np.all(qd == 0.) and np.all(tstates == 0.)))
                                       and (use_ess or
                                            (np.all(pc == 0.) and np.all(pd == 0.) and np.all(estates == 0.))))}
    output['checks'] = checks
    output['verification'] = {
        'tolerance': tol, 'maximum_linear_violation': linerr,
        'maximum_variable_bound_violation': bounderr, 'maximum_binary_mode_error': binerr,
        'maximum_pcc_violation_mw': caperr,
        'tes_terminal_energy_error_mwh_th': terr, 'ess_terminal_energy_error_mwh': eerr,
        'maximum_tes_replay_error_mwh_th': treplay, 'maximum_ess_replay_error_mwh': ereplay,
        'maximum_tes_curtailment_mw_th': curtail,
        'maximum_tes_balance_error_mwh_th': tbalance, 'maximum_ess_balance_error_mwh': ebalance}
    if not all(checks.values()):
        output['status'] = 'verification_failed'
        return output
    output.update(status='optimal' if result.status == 0 else 'feasible_time_limit',
                  success=True, feasible=True, optimal=result.status == 0, rows=replay_rows,
                  schedule={'tes_charge_mw_th': qc.tolist(), 'tes_discharge_mw_th': qd.tolist(),
                            'tes_energy_mwh_th': tstates.tolist(), 'ess_charge_mw': pc.tolist(),
                            'ess_discharge_mw': pd.tolist(), 'ess_energy_mwh': estates.tolist(),
                            'pcc_mw': power.tolist(), 'pcc_upper_mw': caps.tolist(),
                            'no_storage_pcc_mw': p0.tolist()})
    output['metrics'] = {
        'pcc_energy_mwh': float(np.sum(power)*dt), 'no_storage_pcc_energy_mwh': float(np.sum(p0)*dt),
        'pcc_energy_increment_mwh': float(np.sum(power-p0)*dt),
        'pcc_peak_mw': float(np.max(power)), 'no_storage_pcc_peak_mw': float(np.max(p0)),
        'tes_charge_energy_mwh_th': float(np.sum(qc)*dt),
        'tes_discharge_energy_mwh_th': float(np.sum(qd)*dt),
        'ess_charge_energy_mwh': float(np.sum(pc)*dt), 'ess_discharge_energy_mwh': float(np.sum(pd)*dt),
        'tes_initial_energy_mwh_th': float(tstates[0]), 'tes_terminal_energy_mwh_th': float(tstates[-1]),
        'ess_initial_energy_mwh': float(estates[0]), 'ess_terminal_energy_mwh': float(estates[-1]),
        'tes_minimum_soc': float(np.min(tstates)/tcap) if use_tes else None,
        'tes_maximum_soc': float(np.max(tstates)/tcap) if use_tes else None,
        'ess_minimum_soc': float(np.min(estates)/ecap) if use_ess else None,
        'ess_maximum_soc': float(np.max(estates)/ecap) if use_ess else None,
        'maximum_chiller_mw_th': max(row['chiller_mw_th'] for row in replay_rows),
        'tes_standby_loss_mwh_th': sum(row['tes_standby_loss_mwh_th'] for row in replay_rows),
        'ess_standby_loss_mwh': sum(row['ess_standby_loss_mwh'] for row in replay_rows),
        'tes_charge_conversion_loss_mwh_th': float((1-tc['charge_efficiency'])*np.sum(qc)*dt),
        'tes_discharge_conversion_loss_mwh_th': float((1/tc['discharge_efficiency']-1)*np.sum(qd)*dt),
        'ess_charge_conversion_loss_mwh': float((1-ec['charge_efficiency'])*np.sum(pc)*dt),
        'ess_discharge_conversion_loss_mwh': float((1/ec['discharge_efficiency']-1)*np.sum(pd)*dt),
        'maximum_pcc_reduction_mw': float(max(0., np.max(p0-power))),
        'maximum_pcc_increase_mw': float(max(0., np.max(power-p0))),
        'maximum_tes_pcc_reduction_mw': float(max(0., np.max(b*qd-a*qc))),
        'maximum_ess_pcc_reduction_mw': float(max(0., np.max(pd-pc))),
    }
    return output
