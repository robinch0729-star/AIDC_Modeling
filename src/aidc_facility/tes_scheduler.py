"""Deterministic TES scheduling under an exogenous PCC import upper limit.

The upper limit is an input, not a network-derived dynamic operating envelope.
This linear, perfect-information model uses the frozen thermal service demand,
constant COP and capacity-based TES standing loss. It optimizes imported
electrical energy, not tariff cost, and includes paid terminal energy recovery.
No slack variables, post-hoc state resets, or dispatch curtailment are used to
make an infeasible schedule appear compliant. A separate replay through the
existing operating kernel must verify every returned usable schedule.
"""

from collections.abc import Mapping
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .tes_flexibility import marginal_pcc_coefficients, pcc_from_thermal
from .tes_operating import energy_step, validate_tes


def _number(name, value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number, not bool")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _validated_inputs(base_rows, pcc_upper_mw, tes_config, legacy, dt_h,
                      initial_soc, terminal_soc, cop, time_limit_s):
    c = validate_tes(tes_config)
    if c['hourly_retention_fraction'] != 1.0:
        raise ValueError("Scheduler supports capacity-only standing loss; require hourly_retention_fraction=1")
    dt, cop, time_limit = (_number('dt_h', dt_h), _number('cop', cop),
                           _number('time_limit_s', time_limit_s))
    if min(dt, cop, time_limit) <= 0:
        raise ValueError("dt_h, cop and time_limit_s must be positive")
    soc0, socn = _number('initial_soc', initial_soc), _number('terminal_soc', terminal_soc)
    if not (c['min_soc'] <= soc0 <= c['max_soc'] and c['min_soc'] <= socn <= c['max_soc']):
        raise ValueError("Initial and terminal SOC must lie inside TES SOC bounds")
    if not isinstance(legacy, Mapping):
        raise ValueError("legacy must be a mapping")
    plant, electrical = {}, {}
    for group, destination, keys in (
        ('plant', plant, ('chiller_capacity_mw_th', 'condenser_pump_mw_per_mw_heat',
                          'tower_fan_mw_per_mw_heat')),
        ('electrical', electrical, ('ups_efficiency', 'transformer_efficiency', 'auxiliary_mw')),
    ):
        if not isinstance(legacy.get(group), Mapping):
            raise ValueError(f"legacy.{group} must be a mapping")
        for key in keys:
            if key not in legacy[group]:
                raise ValueError(f"Missing legacy.{group}.{key}")
            destination[key] = _number(f'legacy.{group}.{key}', legacy[group][key])
    if plant['chiller_capacity_mw_th'] <= 0:
        raise ValueError("Chiller capacity must be positive")
    if min(plant['condenser_pump_mw_per_mw_heat'], plant['tower_fan_mw_per_mw_heat'],
           electrical['auxiliary_mw']) < 0:
        raise ValueError("Plant coefficients and auxiliary demand must be nonnegative")
    if not all(0 < electrical[k] <= 1 for k in ('ups_efficiency', 'transformer_efficiency')):
        raise ValueError("Electrical efficiencies must lie in (0, 1]")
    try:
        raw_rows = list(base_rows)
    except TypeError as exc:
        raise ValueError("base_rows must be an iterable of mappings") from exc
    if not raw_rows:
        raise ValueError("base_rows must be nonempty")
    rows = []
    for i, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"base_rows[{i}] must be a mapping")
        clean = {}
        for key in ('it_mw', 'fan_mw', 'water_pump_mw', 'chiller_mw_th'):
            if key not in row:
                raise ValueError(f"base_rows[{i}] is missing {key}")
            clean[key] = _number(f'base_rows[{i}].{key}', row[key])
            if clean[key] < 0:
                raise ValueError(f"base_rows[{i}].{key} must be nonnegative")
        if clean['chiller_mw_th'] > plant['chiller_capacity_mw_th']:
            raise ValueError("Frozen baseline exceeds chiller capacity; existing replay kernel does not support this case")
        rows.append(clean)
    if np.isscalar(pcc_upper_mw):
        caps = np.full(len(rows), _number('pcc_upper_mw', pcc_upper_mw))
    else:
        try:
            values = list(pcc_upper_mw)
        except TypeError as exc:
            raise ValueError("pcc_upper_mw must be a scalar or one-dimensional sequence") from exc
        if len(values) != len(rows):
            raise ValueError("PCC upper-limit length must match base_rows")
        caps = np.array([_number(f'pcc_upper_mw[{i}]', value) for i, value in enumerate(values)])
    if np.any(caps < 0):
        raise ValueError("PCC import upper limits must be nonnegative")
    return rows, caps, c, {'plant': plant, 'electrical': electrical}, dt, soc0, socn, cop, time_limit


def solve_schedule(base_rows, pcc_upper_mw, tes_config, legacy, dt_h=1/12,
                   initial_soc=.5, terminal_soc=.5, cop=5., objective='energy',
                   time_limit_s=60):
    """Minimize total PCC electricity subject to hard 5-minute-style limits.

    ``base_rows`` contains one frozen mean thermal/electrical demand per bin:
    it_mw, fan_mw, water_pump_mw, chiller_mw_th. ``pcc_upper_mw`` may be one
    scalar or a same-length sequence. Both initial and terminal SOC constraints
    are enforced; callers should explicitly set them for fair comparisons.

    Returned ``feasible`` requires independent physical replay. ``optimal``
    additionally requires the solver's optimal status (within its reported
    numerical tolerance). A verified incumbent at the time limit is labelled
    ``feasible_time_limit``, never optimal. Infeasible runs return no schedule.
    All returned values are JSON serializable, without NaN or infinity.
    """
    if objective != 'energy':
        raise ValueError("Only objective='energy' is implemented; no tariff or peak objective is implied")
    rows, caps, c, legacy, dt, soc0, socn, cop, time_limit = _validated_inputs(
        base_rows, pcc_upper_mw, tes_config, legacy, dt_h, initial_soc, terminal_soc, cop, time_limit_s)
    n = len(rows)
    capacity = c['energy_capacity_mwh_th']
    emin, emax = c['min_soc'] * capacity, c['max_soc'] * capacity
    e0, en = soc0 * capacity, socn * capacity
    qcmax, qdmax = c['charge_limit_mw_th'], c['discharge_limit_mw_th']
    eta_c, eta_d = c['charge_efficiency'], c['discharge_efficiency']
    pump = c['pump_mw_per_mw_th']
    loss = c['capacity_loss_fraction_per_hour'] * capacity
    chill_capacity = legacy['plant']['chiller_capacity_mw_th']
    p0 = np.array([pcc_from_thermal(r, 0., 0., legacy, cop, pump)['pcc_mw'] for r in rows])
    marginal = marginal_pcc_coefficients(legacy, cop, pump)
    a, b = marginal['increase_mwe_per_mwth'], marginal['reduction_mwe_per_mwth']
    # Variables: Qcharge[n], Qdischarge[n], E[n+1], mode[n].
    ie, iz, nv = 2*n, 3*n+1, 4*n+1
    lower = np.zeros(nv)
    upper = np.concatenate((np.full(n, qcmax), np.full(n, qdmax),
                            np.full(n+1, emax), np.ones(n)))
    lower[ie:ie+n+1] = emin
    lower[ie] = upper[ie] = e0
    lower[ie+n] = upper[ie+n] = en
    integrality = np.zeros(nv, dtype=np.uint8)
    integrality[iz:] = 1
    objective_coeff = np.zeros(nv)
    objective_coeff[:n] = a * dt
    objective_coeff[n:2*n] = -b * dt
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
        constraint(((ie+k+1, 1.), (ie+k, -1.), (k, -dt*eta_c), (n+k, dt/eta_d)),
                   -dt*loss, -dt*loss)
        constraint(((k, 1.), (iz+k, -qcmax)), -np.inf, 0.)
        constraint(((n+k, 1.), (iz+k, qdmax)), -np.inf, qdmax)
        constraint(((k, 1+pump), (n+k, -(1-pump))),
                   -base['chiller_mw_th'], chill_capacity-base['chiller_mw_th'])
        constraint(((k, a), (n+k, -b)), -p0[k], caps[k]-p0[k])
    matrix = coo_matrix((values, (ri, ci)), shape=(len(lb), nv)).tocsc()
    result = milp(objective_coeff, integrality=integrality, bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(matrix, np.array(lb), np.array(ub)),
                  options={'time_limit': time_limit, 'mip_rel_gap': 1e-8, 'presolve': True})
    status_names = {0: 'optimal', 1: 'time_limit', 2: 'infeasible', 3: 'unbounded', 4: 'solver_error'}

    def finite_optional(value):
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    solver = {'status_code': int(result.status), 'message': str(result.message),
              'success': bool(result.success), 'mip_gap': finite_optional(getattr(result, 'mip_gap', None)),
              'mip_node_count': finite_optional(getattr(result, 'mip_node_count', None)),
              'objective_increment_mwh': finite_optional(getattr(result, 'fun', None)),
              'dual_bound_increment_mwh': finite_optional(getattr(result, 'mip_dual_bound', None)),
              'time_limit_s': time_limit}
    output = {'status': status_names.get(result.status, 'solver_error'), 'success': False,
              'feasible': False, 'optimal': False, 'objective': objective,
              'solver': solver, 'rows': [], 'schedule': None, 'checks': {}, 'metrics': None,
              'contract': {'limit_source': 'exogenous_input_not_network_derived',
                           'perfect_information': True, 'hard_pcc_limits': True,
                           'slack_variables': False, 'dt_h': dt, 'intervals': n,
                           'initial_soc': soc0, 'terminal_soc': socn, 'cop': cop,
                           'capacity_standby_loss_mw_th': loss,
                           'pcc_increase_mwe_per_mwth': a, 'pcc_reduction_mwe_per_mwth': b}}
    if result.status not in (0, 1) or result.x is None:
        return output
    x = np.asarray(result.x, dtype=float)
    if x.shape != (nv,) or not np.all(np.isfinite(x)):
        output['status'] = 'verification_failed'
        output['verification_error'] = 'Nonfinite or wrong-size solver incumbent'
        return output
    tol = 1e-7
    qc, qd = x[:n].copy(), x[n:2*n].copy()
    # Only remove floating-point signed zero; substantive corrections forbidden.
    for ports in (qc, qd):
        ports[np.abs(ports) < 1e-10] = 0.
    replay_rows, energy, states = [], e0, [e0]
    try:
        for k, base in enumerate(rows):
            if qc[k] < 0 or qd[k] < 0 or qc[k]*qd[k] > 1e-12:
                raise ValueError('Negative or simultaneous charge/discharge in incumbent')
            step = energy_step(energy, float(qd[k]-qc[k]), dt,
                               base['chiller_mw_th'], chill_capacity, c)
            power = pcc_from_thermal(base, float(qc[k]), float(qd[k]), legacy, cop, pump)
            energy = step['energy_end_mwh_th']
            states.append(energy)
            replay_rows.append(dict(step, source_index=k, time_h=k*dt,
                                    command_charge_mw_th=float(qc[k]), command_discharge_mw_th=float(qd[k]),
                                    pcc_mw=power['pcc_mw'], pcc_upper_mw=float(caps[k]),
                                    no_tes_pcc_mw=float(p0[k]), delta_pcc_mw=power['pcc_mw']-float(p0[k]),
                                    base_chiller_mw_th=base['chiller_mw_th']))
    except ValueError as exc:
        output['status'] = 'verification_failed'
        output['verification_error'] = str(exc)
        return output
    power = np.array([r['pcc_mw'] for r in replay_rows])
    states = np.array(states)
    linear_value = matrix @ x
    lo, hi = np.array(lb), np.array(ub)
    linear_violation = max(float(np.max(np.maximum(lo-linear_value, 0.))),
                           float(np.max(np.maximum(linear_value-hi, 0.))))
    bound_violation = max(float(np.max(np.maximum(lower-x, 0.))),
                          float(np.max(np.maximum(x-upper, 0.))))
    replay_error = float(np.max(np.abs(states-x[ie:ie+n+1])))
    curtailment = max(r['curtailed_mw_th'] for r in replay_rows)
    balance_error = max(abs(r['energy_balance_error_mwh_th']) for r in replay_rows)
    cap_violation = float(np.max(np.maximum(power-caps, 0.)))
    terminal_error = abs(float(states[-1])-en)
    checks = {
        'linear_constraints': linear_violation <= tol,
        'variable_bounds': bound_violation <= tol,
        'binary_modes': float(np.max(np.abs(x[iz:]-np.rint(x[iz:])))) <= tol,
        'mutually_exclusive_ports': bool(np.all(qc*qd <= 1e-12)),
        'pcc_upper_limit': cap_violation <= tol,
        'pcc_nonnegative': bool(np.min(power) >= -tol),
        'storage_bounds': bool(np.min(states) >= emin-tol and np.max(states) <= emax+tol),
        'terminal_energy': terminal_error <= tol,
        'chiller_capacity': all(-tol <= r['chiller_mw_th'] <= chill_capacity+tol for r in replay_rows),
        'no_dispatch_curtailment': curtailment <= tol,
        'independent_energy_replay': replay_error <= tol,
        'energy_balance': balance_error <= tol,
        'pcc_linear_equivalence': bool(np.max(np.abs(power-(p0+a*qc-b*qd))) <= tol),
    }
    output['checks'] = checks
    output['verification'] = {'tolerance': tol, 'maximum_linear_violation': linear_violation,
                              'maximum_variable_bound_violation': bound_violation,
                              'maximum_pcc_violation_mw': cap_violation,
                              'terminal_energy_error_mwh_th': terminal_error,
                              'maximum_replay_energy_error_mwh_th': replay_error,
                              'maximum_curtailment_mw_th': curtailment,
                              'maximum_energy_balance_error_mwh_th': balance_error}
    if not all(checks.values()):
        output['status'] = 'verification_failed'
        return output
    output.update(status='optimal' if result.status == 0 else 'feasible_time_limit',
                  success=True, feasible=True, optimal=result.status == 0, rows=replay_rows,
                  schedule={'charge_mw_th': qc.tolist(), 'discharge_mw_th': qd.tolist(),
                            'energy_mwh_th': states.tolist(), 'pcc_mw': power.tolist(),
                            'pcc_upper_mw': caps.tolist(), 'no_tes_pcc_mw': p0.tolist()})
    output['metrics'] = {
        'pcc_energy_mwh': float(np.sum(power)*dt),
        'no_tes_pcc_energy_mwh': float(np.sum(p0)*dt),
        'pcc_energy_increment_mwh': float(np.sum(power-p0)*dt),
        'pcc_peak_mw': float(np.max(power)), 'no_tes_pcc_peak_mw': float(np.max(p0)),
        'charge_energy_mwh_th': float(np.sum(qc)*dt),
        'discharge_energy_mwh_th': float(np.sum(qd)*dt),
        'initial_energy_mwh_th': float(states[0]), 'terminal_energy_mwh_th': float(states[-1]),
        'minimum_soc': float(np.min(states)/capacity), 'maximum_soc': float(np.max(states)/capacity),
        'maximum_chiller_mw_th': max(r['chiller_mw_th'] for r in replay_rows),
        'standby_loss_mwh_th': loss*n*dt,
        'charge_conversion_loss_mwh_th': float((1-eta_c)*np.sum(qc)*dt),
        'discharge_conversion_loss_mwh_th': float((1/eta_d-1)*np.sum(qd)*dt),
        'maximum_pcc_reduction_mw': float(max(0., np.max(p0-power))),
        'maximum_pcc_increase_mw': float(max(0., np.max(power-p0))),
    }
    return output
