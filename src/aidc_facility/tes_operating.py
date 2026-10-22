"""Reduced chilled-water TES operating model, with explicit thermal-energy units.

This is an operating abstraction, not a tank/coil or water-temperature design.
The caller supplies a literature/scenario parameter set. Charge and discharge
are thermal rates at the cooling-loop boundary in MW_th; stored energy is in
MWh_th. A fixed usable energy interval represents the adopted cooling operating
window. No claim of outlet-temperature, flow, or heat-exchanger validation is
made. Charge/discharge conversion losses and standby loss are separate. Standby
loss may be proportional to stored energy, a constant fraction of rated energy
capacity per hour, or both; sources using these definitions are not interchangeable.

Positive requested net power means discharge; negative means charge. TES pump
electricity becomes heat in the cooling loop. With pump coefficient k,
Q_chiller = Q_base + k*(Q_charge + Q_discharge) + Q_charge - Q_discharge.
The coefficient is an operating assumption, not a pump hydraulic design.
"""

import math
from collections.abc import Mapping


_KEYS = (
    "energy_capacity_mwh_th",
    "min_soc",
    "max_soc",
    "charge_limit_mw_th",
    "discharge_limit_mw_th",
    "charge_efficiency",
    "discharge_efficiency",
    "hourly_retention_fraction",
    "pump_mw_per_mw_th",
)


def _finite(name, value):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not bool")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def validate_tes(config):
    """Validate and return a numeric copy, without mutating caller parameters."""
    if not isinstance(config, Mapping):
        raise ValueError("tes_config must be a mapping")
    missing = [key for key in _KEYS if key not in config]
    if missing:
        raise ValueError("Missing TES parameters: " + ", ".join(missing))
    c = {key: _finite(key, config[key]) for key in _KEYS}
    c["capacity_loss_fraction_per_hour"] = _finite(
        "capacity_loss_fraction_per_hour", config.get("capacity_loss_fraction_per_hour", 0.0)
    )
    if c["energy_capacity_mwh_th"] <= 0:
        raise ValueError("energy_capacity_mwh_th must be positive")
    if not 0 <= c["min_soc"] < c["max_soc"] <= 1:
        raise ValueError("Require 0 <= min_soc < max_soc <= 1")
    for key in ("charge_limit_mw_th", "discharge_limit_mw_th"):
        if c[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    for key in ("charge_efficiency", "discharge_efficiency", "hourly_retention_fraction"):
        if not 0 < c[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if not 0 <= c["pump_mw_per_mw_th"] < 1:
        raise ValueError("pump_mw_per_mw_th must be in [0, 1)")
    if c["capacity_loss_fraction_per_hour"] < 0:
        raise ValueError("capacity_loss_fraction_per_hour must be nonnegative")
    return c


def _coefficients(dt_h, c):
    dt_h = _finite("dt_h", dt_h)
    if dt_h <= 0:
        raise ValueError("dt_h must be positive")
    decay = -math.log(c["hourly_retention_fraction"])
    if decay == 0:
        return dt_h, 1.0, dt_h, decay
    exponent = -decay * dt_h
    return dt_h, math.exp(exponent), -math.expm1(exponent) / decay, decay


def _energy_bounds(energy, c):
    energy = _finite("energy_mwh_th", energy)
    capacity = c["energy_capacity_mwh_th"]
    low, high = capacity * c["min_soc"], capacity * c["max_soc"]
    tolerance = 1e-10 * max(1.0, capacity)
    if energy < 0 or energy < low - tolerance or energy > high + tolerance:
        raise ValueError("Initial stored energy lies outside configured SOC bounds")
    return energy, low, high, tolerance


def energy_step(energy_mwh_th, requested_net_mw_th, dt_h,
                base_chiller_mw_th, chiller_capacity_mw_th, tes_config):
    """Advance one constant-command interval using the exact scalar solution.

    dE/dt = -lambda*E + eta_charge*Q_charge - Q_discharge/eta_discharge
            - capacity_loss_fraction_per_hour*energy_capacity_mwh_th.
    lambda = -log(hourly_retention_fraction), so changing timestep does not
    change the specified hourly standby retention. Dispatch is curtailed by
    energy endpoints, equipment ratings, and the cooling-loop chiller balance.

    An infeasible base cooling demand raises ValueError. If standby loss alone
    crosses the minimum SOC, the actual endpoint and violation are returned;
    stored energy is never reset to the minimum or replenished implicitly.
    A capacity-based loss model extrapolated below zero energy is physically
    invalid. Such an endpoint is returned with physical_energy_violation=True,
    rather than silently clipped. Starting a later step outside configured
    bounds raises ValueError, so callers must prevent or handle that failure.
    """
    c = validate_tes(tes_config)
    energy, low, high, tolerance = _energy_bounds(energy_mwh_th, c)
    requested = _finite("requested_net_mw_th", requested_net_mw_th)
    base = _finite("base_chiller_mw_th", base_chiller_mw_th)
    capacity = _finite("chiller_capacity_mw_th", chiller_capacity_mw_th)
    if base < 0 or capacity < 0:
        raise ValueError("Chiller demand and capacity must be nonnegative")
    if base > capacity:
        raise ValueError("Base chiller demand exceeds chiller capacity")
    dt, a, b, decay = _coefficients(dt_h, c)
    capacity_loss_rate = c["capacity_loss_fraction_per_hour"] * c["energy_capacity_mwh_th"]
    idle_end = a * energy - b * capacity_loss_rate
    eta_c, eta_d = c["charge_efficiency"], c["discharge_efficiency"]
    k = c["pump_mw_per_mw_th"]
    charge = discharge = 0.0

    if requested < 0:
        upper_energy_rate = max(0.0, (high - idle_end) / (eta_c * b))
        headroom_rate = max(0.0, (capacity - base) / (1 + k))
        charge = min(-requested, c["charge_limit_mw_th"], upper_energy_rate, headroom_rate)
    elif requested > 0:
        lower_energy_rate = max(0.0, (idle_end - low) * eta_d / b)
        cooling_demand_rate = base / (1 - k)
        discharge = min(requested, c["discharge_limit_mw_th"], lower_energy_rate, cooling_demand_rate)

    stored_input = eta_c * charge - discharge / eta_d
    end = idle_end + b * stored_input
    capacity_loss = capacity_loss_rate * dt
    # Exact integral of lambda*E; the zero-lambda branch avoids cancellation.
    proportional_loss = 0.0 if decay == 0 else energy - end + (stored_input - capacity_loss_rate) * dt
    if proportional_loss < 0 and proportional_loss >= -tolerance:
        proportional_loss = 0.0
    standby = proportional_loss + capacity_loss
    charge_loss = (1 - eta_c) * charge * dt
    discharge_loss = (1 / eta_d - 1) * discharge * dt
    pump = k * (charge + discharge)
    chiller = base + pump + charge - discharge
    actual = discharge - charge
    balance_error = end - energy - ((charge - discharge) * dt - charge_loss - discharge_loss - standby)
    lower_violation, upper_violation = max(0.0, low - end), max(0.0, end - high)
    return {
        "energy_start_mwh_th": energy,
        "energy_end_mwh_th": end,
        "requested_net_mw_th": requested,
        "actual_net_mw_th": actual,
        "charge_mw_th": charge,
        "discharge_mw_th": discharge,
        "pump_mw": pump,
        "chiller_mw_th": chiller,
        "standby_loss_mwh_th": standby,
        "proportional_standby_loss_mwh_th": proportional_loss,
        "capacity_loss_mwh_th": capacity_loss,
        "charge_conversion_loss_mwh_th": charge_loss,
        "discharge_conversion_loss_mwh_th": discharge_loss,
        "curtailed_mw_th": max(0.0, abs(requested) - abs(actual)),
        "state_violation": lower_violation > tolerance or upper_violation > tolerance,
        "physical_energy_violation": end < -tolerance,
        "negative_energy_mwh_th": max(0.0, -end),
        "lower_bound_violation_mwh_th": lower_violation,
        "upper_bound_violation_mwh_th": upper_violation,
        "energy_balance_error_mwh_th": balance_error,
    }


def target_charge_request(energy_mwh_th, target_mwh_th, dt_h, tes_config):
    """Return an unbounded negative net request to reach a target by charging.

    The energy_step function must still enforce plant and storage limits; this
    helper never overwrites the state. If the target is below the idle endpoint,
    return zero (this is a charge-only helper, not a discharge controller).
    """
    c = validate_tes(tes_config)
    energy, low, high, tolerance = _energy_bounds(energy_mwh_th, c)
    target = _finite("target_mwh_th", target_mwh_th)
    if target < low - tolerance or target > high + tolerance:
        raise ValueError("Target stored energy lies outside configured SOC bounds")
    _, a, b, _ = _coefficients(dt_h, c)
    capacity_loss_rate = c["capacity_loss_fraction_per_hour"] * c["energy_capacity_mwh_th"]
    idle_end = a * energy - b * capacity_loss_rate
    return -max(0.0, (target - idle_end) / (c["charge_efficiency"] * b))
