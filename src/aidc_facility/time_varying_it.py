"""Fixed-node, workload-driven M3+R3 research prototype.

References: NLR [3] Sec. 2.3.2 motivates offered request allocation and
bottom-up aggregation; [43] Sec. 3 motivates arrival-driven power generation.
Neither paper supplies this M3+R3 composition. This module keeps the locked
M3/R3 coefficients and extends their *fixed-condition* meaning quasi-statically.
It does not validate instantaneous rate transitions, queueing or autoscaling.

Crucially, residual state survives traffic boundaries; no finite-window
demeaning, state reset, clipping of power, or post-generation scaling is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.signal import lfilter

from nlr_data.final_it_profile_generator import FinalITProfileGenerator


@dataclass(frozen=True)
class FixedService:
    key: str
    prompt_dataset: str
    node_count: int
    reference_rate_rps: float
    # This is an input-support/admission guard, NOT a validated throughput/SLO.
    supported_max_rate_rps: float = 100.0

    def validate(self) -> None:
        if not self.key or not self.prompt_dataset:
            raise ValueError("Service key and prompt dataset must be nonempty")
        if isinstance(self.node_count, bool) or not isinstance(self.node_count, Integral) or self.node_count < 1:
            raise ValueError("node_count must be a positive integer")
        if not np.isfinite(self.reference_rate_rps) or self.reference_rate_rps <= 0:
            raise ValueError("reference rate must be positive")
        if not np.isfinite(self.supported_max_rate_rps) or self.supported_max_rate_rps <= 0:
            raise ValueError("supported rate guard must be positive")


@dataclass(frozen=True)
class R3Coefficients:
    phi: float
    jump_gate: float
    gaussian_std_kw: float
    weights: np.ndarray
    means_kw: np.ndarray
    stds_kw: np.ndarray

    @property
    def innovation_variance(self) -> float:
        mixture = float(np.dot(self.weights, self.stds_kw**2 + self.means_kw**2))
        return self.jump_gate * mixture + (1.0 - self.jump_gate) * self.gaussian_std_kw**2

    @property
    def stationary_std_kw(self) -> float:
        return math.sqrt(self.innovation_variance / (1.0 - self.phi**2))


def coefficients_at(generator: FinalITProfileGenerator, dataset: str, rate: float) -> R3Coefficients:
    """Reuse the existing eight-neighbour interpolation, without refitting."""
    if generator.request_rate_status(dataset, rate) == "unsupported_extrapolation":
        raise ValueError(f"Unsupported rate {rate:g} for {dataset}; idle is not inferred from M3")
    lib = generator.residual_library
    get = lambda name: lib.weighted(dataset, rate, name)
    weights = np.maximum([get(f"{role}_weight") for role in ("negative", "background", "positive")], 1e-8)
    weights = weights / weights.sum()
    means = np.array([get(f"{role}_mean_kw") for role in ("negative", "background", "positive")])
    means -= np.dot(weights, means)
    stds = np.maximum([get(f"{role}_std_kw") for role in ("negative", "background", "positive")], 1e-12)
    result = R3Coefficients(
        phi=float(np.clip(get("phi"), 0.0, 0.9995)),
        jump_gate=float(np.clip(get("jump_evidence"), 0.0, 1.0)),
        gaussian_std_kw=max(float(get("innovation_std_kw")), 1e-12),
        weights=weights, means_kw=means, stds_kw=stds,
    )
    values = [result.phi, result.jump_gate, result.gaussian_std_kw, *weights, *means, *stds]
    if not np.isfinite(values).all():
        raise ValueError("Non-finite locked residual coefficients")
    return result


def advance_residual(
    coefficients: R3Coefficients,
    previous_state_kw: np.ndarray,
    steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """e[k] = phi*e[k-1] + eta[k], carrying state across traffic blocks.

    The innovation distribution is unchanged from the locked R3 generator.
    Unlike the legacy finite-window helper, samples are NOT demeaned.
    Returned arrays are (unit, time) and final state (unit,).
    """
    state = np.asarray(previous_state_kw, dtype=float)
    if state.ndim != 1 or not state.size or not np.isfinite(state).all() or steps < 1:
        raise ValueError("Finite 1-D state and positive steps required")
    shape = (state.size, steps)
    u = rng.random(shape)
    component = np.searchsorted(np.cumsum(coefficients.weights), u, side="right")
    component = np.minimum(component, 2)
    mixture = rng.normal(coefficients.means_kw[component], coefficients.stds_kw[component])
    gate = rng.random(shape) < coefficients.jump_gate
    gaussian = rng.normal(0.0, coefficients.gaussian_std_kw, size=shape)
    innovation = np.where(gate, mixture, gaussian)
    values, _ = lfilter(
        [1.0], [1.0, -coefficients.phi], innovation, axis=1,
        zi=(coefficients.phi * state)[:, None],
    )
    return values, values[:, -1].copy()


def reference_design_scale(
    generator: FinalITProfileGenerator,
    services: Sequence[FixedService],
    target_reference_it_mw: float,
) -> float:
    """Choose offered-load level once at the constant reference design point.

    Node counts and relative reference rates are fixed. This changes input
    demand BEFORE simulation, never rescales any output realization.
    """
    if not services or len({s.key for s in services}) != len(services):
        raise ValueError("Nonempty services with unique keys required")
    for service in services:
        service.validate()
    low = max(generator.rate_ranges(s.prompt_dataset)["calibration"][0] / s.reference_rate_rps for s in services)
    high = min(s.supported_max_rate_rps / s.reference_rate_rps for s in services)
    def objective(scale: float) -> float:
        return sum(s.node_count * float(generator.m3_parameters[s.prompt_dataset].predict(s.reference_rate_rps * scale)) for s in services) / 1000.0 - target_reference_it_mw
    if low >= high or objective(low) * objective(high) > 0:
        raise ValueError("Target reference mean cannot be reached within supported fixed-node rates")
    return float(brentq(objective, low, high))


def make_dispatch(
    traffic: pd.DataFrame,
    services: Sequence[FixedService],
    count_columns: dict[str, str],
    rate_scale: float,
    *,
    variable: bool,
) -> pd.DataFrame:
    """Rescale source *arrival intensity*, preserving native elapsed time.

    Each service trace is normalized by its own period mean, then multiplied
    by a documented scenario demand level. This is NOT measured NLR traffic,
    nor a claim about actual Azure facility power/absolute workload size.
    Excess requests are recorded, not silently discarded or called completed.
    """
    required = {"time_s", "bin_s", *count_columns.values()}
    if not required.issubset(traffic):
        raise ValueError(f"Missing traffic columns: {sorted(required.difference(traffic))}")
    if len(traffic) < 2 or not np.isfinite(rate_scale) or rate_scale <= 0:
        raise ValueError("At least two traffic bins and positive rate scale required")
    widths = traffic["bin_s"].to_numpy(float)
    times = traffic["time_s"].to_numpy(float)
    if not np.isfinite(widths).all() or np.any(widths <= 0) or not np.allclose(widths, widths[0]):
        raise ValueError("Positive equal-width bins required")
    if not np.isfinite(times).all() or not np.isclose(times[0], 0.0) or not np.allclose(np.diff(times), widths[:-1]):
        raise ValueError("Contiguous traffic bins must start at time zero")
    rows: list[dict] = []
    for service in services:
        service.validate()
        counts = traffic[count_columns[service.key]].to_numpy(float)
        if not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError("Arrival counts must be finite and nonnegative")
        observed_rates = counts / widths
        mean_source_rate = float(np.sum(counts) / np.sum(widths))
        if mean_source_rate <= 0:
            raise ValueError(f"No source arrivals for {service.key}")
        target_per_node = service.reference_rate_rps * rate_scale
        traffic_multiplier = service.node_count * target_per_node / mean_source_rate
        offered = (observed_rates if variable else np.full(len(traffic), mean_source_rate)) * traffic_multiplier
        maximum = service.node_count * service.supported_max_rate_rps
        admitted = np.minimum(offered, maximum)
        for index in range(len(traffic)):
            rows.append({
                "service": service.key, "prompt_dataset": service.prompt_dataset,
                "time_s": float(times[index]), "bin_s": float(widths[index]),
                "node_count": service.node_count,
                "source_count": float(counts[index]),
                "source_arrival_rate_rps": float(observed_rates[index]),
                "source_to_scenario_intensity_multiplier": traffic_multiplier,
                "offered_rps": float(offered[index]), "admitted_rps": float(admitted[index]),
                "excess_rps": float(offered[index] - admitted[index]),
                "node_offered_arrival_rps": float(admitted[index] / service.node_count),
                "input_support_guard_rps_per_node": service.supported_max_rate_rps,
                "completed_throughput_modeled": False,
            })
    result = pd.DataFrame(rows)
    if not np.allclose(result.offered_rps, result.admitted_rps + result.excess_rps):
        raise AssertionError("Request conservation failed")
    return result


def validate_dispatch(generator: FinalITProfileGenerator, dispatch: pd.DataFrame, interval_s: float) -> dict:
    if not np.isclose(interval_s, generator.calibrated_interval_s):
        raise ValueError("Only the calibrated residual time step is allowed")
    issues: list[dict] = []
    statuses: dict[str, int] = {}
    for row in dispatch.itertuples(index=False):
        if not np.isclose(row.bin_s / interval_s, round(row.bin_s / interval_s)):
            raise ValueError("Traffic bins must contain an integer number of residual steps")
        status = generator.request_rate_status(row.prompt_dataset, row.node_offered_arrival_rps)
        statuses[status] = statuses.get(status, 0) + 1
        if status == "unsupported_extrapolation":
            issues.append({"service": row.service, "time_s": row.time_s, "rate": row.node_offered_arrival_rps})
    if issues:
        raise ValueError(f"Unsupported/zero-load bins; no automatic idle substitute: {issues[:5]}")
    return {"rate_support_counts": statuses, "unsupported_bins": 0,
            "excess_requests": float(np.sum(dispatch.excess_rps * dispatch.bin_s))}


def simulate_case(
    generator: FinalITProfileGenerator,
    dispatch: pd.DataFrame,
    services: Sequence[FixedService],
    *,
    case: str,
    scenario_count: int = 5,
    interval_s: float = 0.1,
    seed: int = 29092026,
    burn_in_s: float = 120.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Direct independent node sum; common variation enters through traffic.

    Shared seeds across matched cases provide common random numbers, not
    independent validation. One arrival trace remains one demand observation.
    """
    if not services or len({s.key for s in services}) != len(services):
        raise ValueError("Nonempty services with unique keys required")
    if set(dispatch.service) != {s.key for s in services}:
        raise ValueError("Dispatch services do not match service specifications")
    for service in services:
        service.validate()
        part = dispatch.loc[dispatch.service.eq(service.key)]
        if not part.node_count.eq(service.node_count).all() or not part.prompt_dataset.eq(service.prompt_dataset).all():
            raise ValueError("Dispatch node count/dataset does not match service specifications")
    validate_dispatch(generator, dispatch, interval_s)
    if scenario_count < 1 or not np.isfinite(burn_in_s) or burn_in_s < 0:
        raise ValueError("Positive scenario count and nonnegative burn-in required")
    reference = dispatch.loc[dispatch.service.eq(services[0].key)].sort_values("time_s")
    length = int(round(float(reference.bin_s.sum()) / interval_s))
    time_s = np.arange(length) * interval_s
    bin_starts = np.rint(reference.time_s.to_numpy() / interval_s).astype(int)
    boundary = np.zeros(length, dtype=bool)
    boundary[bin_starts[1:]] = True
    frames, node_frames, audits = [], [], []
    for scenario in range(scenario_count):
        total = np.zeros(length)
        steady_total = np.zeros(length)
        component_columns = {}
        for service_index, service in enumerate(services):
            rows = dispatch.loc[dispatch.service.eq(service.key)].sort_values("time_s")
            if len(rows) != len(reference) or not np.allclose(rows.time_s, reference.time_s) or not np.allclose(rows.bin_s, reference.bin_s):
                raise ValueError("All services must share the same timeline")
            rng = np.random.default_rng(np.random.SeedSequence([seed, scenario, service_index]))
            first_rate = float(rows.node_offered_arrival_rps.iloc[0])
            first_coeff = coefficients_at(generator, service.prompt_dataset, first_rate)
            state = rng.normal(0.0, first_coeff.stationary_std_kw, size=service.node_count)
            # Approximate stationary initialization followed by unrecorded warmup.
            warm_steps = int(round(burn_in_s / interval_s))
            for start in range(0, warm_steps, 100):
                _, state = advance_residual(first_coeff, state, min(100, warm_steps - start), rng)
            summed = np.empty(length)
            steady = np.empty(length)
            example = np.empty(length)
            residual_example = np.empty(length)
            rate_series = np.empty(length)
            minimum, maximum = np.inf, -np.inf
            over_nominal = 0
            offset = 0
            for row in rows.itertuples(index=False):
                rate = float(row.node_offered_arrival_rps)
                coeff = coefficients_at(generator, service.prompt_dataset, rate)
                steps = int(round(row.bin_s / interval_s))
                point = float(generator.m3_parameters[service.prompt_dataset].predict(rate))
                # Bound temporary memory even for long source bins. This
                # implementation's chunk size is fixed for reproducibility.
                for local_start in range(0, steps, 100):
                    chunk_steps = min(100, steps - local_start)
                    residual, state = advance_residual(coeff, state, chunk_steps, rng)
                    minimum = min(minimum, point + float(residual.min()))
                    maximum = max(maximum, point + float(residual.max()))
                    # [3] hardware nameplate reference, not a hard electrical
                    # cap for the upstream aggregate (CPU domains unresolved).
                    over_nominal += int(np.count_nonzero(point + residual > 3.52))
                    sl = slice(offset + local_start, offset + local_start + chunk_steps)
                    summed[sl] = (service.node_count * point + residual.sum(axis=0)) / 1000.0
                    steady[sl] = service.node_count * point / 1000.0
                    example[sl] = point + residual[0]
                    residual_example[sl] = residual[0]
                    rate_series[sl] = rate
                offset += steps
            if minimum < 0:
                raise ValueError("Negative generated node power: no silent clipping")
            total += summed
            steady_total += steady
            component_columns[f"{service.key}_it_mw"] = summed
            audits.append({"case": case, "scenario_id": scenario, "service": service.key,
                           "minimum_node_reported_power_kw": minimum, "maximum_node_reported_power_kw": maximum,
                           "node_count": service.node_count,
                           "nominal_cpu_gpu_reference_kw": 3.52,
                           "samples_above_nominal_reference": over_nominal,
                           "node_time_samples": service.node_count * length})
            if scenario == 0:
                node_frames.append(pd.DataFrame({"case": case, "service": service.key, "time_s": time_s,
                                                "node_arrival_rate_rps": rate_series,
                                                "example_node_power_kw": example,
                                                "example_residual_kw": residual_example}))
        frames.append(pd.DataFrame({"case": case, "scenario_id": scenario, "time_s": time_s,
                                    "traffic_boundary": boundary, "total_it_mw": total,
                                    "m3_total_mw": steady_total, "residual_total_mw": total - steady_total,
                                    **component_columns}))
    return pd.concat(frames, ignore_index=True), pd.concat(node_frames, ignore_index=True), pd.DataFrame(audits)


def profile_metrics(frame: pd.DataFrame, interval_s: float = 0.1, traffic_bin_s: float = 10.0) -> dict:
    """Report raw and boundary-excluded ramps separately, not a composite J."""
    p = frame.total_it_mw.to_numpy(float)
    if len(p) < 3 or not np.isfinite(p).all():
        raise ValueError("Finite profile with >=3 samples required")
    result = {
        "mean_it_mw": float(p.mean()), "std_it_mw": float(p.std()),
        "cv_percent": float(100 * p.std() / p.mean()),
        "min_it_mw": float(p.min()), "max_it_mw": float(p.max()),
        "p01_it_mw": float(np.quantile(p, 0.01)), "p99_it_mw": float(np.quantile(p, 0.99)),
    }
    boundaries = frame.traffic_boundary.to_numpy(bool) if "traffic_boundary" in frame else np.zeros(len(p), bool)
    prefix = np.cumsum(boundaries.astype(int))
    for horizon in (0.1, 1.0):
        lag = int(round(horizon / interval_s))
        if lag < 1 or lag >= len(p):
            continue
        ramp = (p[lag:] - p[:-lag]) / horizon
        crosses = (prefix[lag:] - prefix[:-lag]) > 0
        name = str(horizon).replace(".", "p")
        result[f"p95_abs_ramp_{name}s_mw_per_s"] = float(np.quantile(np.abs(ramp), 0.95))
        result[f"max_abs_ramp_{name}s_mw_per_s"] = float(np.max(np.abs(ramp)))
        result[f"boundary_crossing_fraction_{name}s"] = float(crosses.mean())
        within = np.abs(ramp[~crosses])
        result[f"p95_within_bin_abs_ramp_{name}s_mw_per_s"] = float(np.quantile(within, 0.95)) if within.size else np.nan
        result[f"max_within_bin_abs_ramp_{name}s_mw_per_s"] = float(within.max()) if within.size else np.nan
    for lag_s in (0.1, 1.0, 10.0):
        lag = int(round(lag_s / interval_s))
        x, y = p[:-lag], p[lag:]
        result[f"acf_{str(lag_s).replace('.', 'p')}s"] = float(np.corrcoef(x, y)[0, 1]) if len(y) > 2 and x.std() > 0 and y.std() > 0 else np.nan
    block = int(round(traffic_bin_s / interval_s))
    if block > 0 and len(p) % block == 0:
        means = p.reshape(-1, block).mean(axis=1)
        if len(means) > 1:
            # Consecutive bin-MEAN difference / bin width; not subsecond ramp.
            result["p95_abs_traffic_bin_mean_ramp_mw_per_s"] = float(np.quantile(np.abs(np.diff(means)) / traffic_bin_s, 0.95))
    if "m3_total_mw" in frame:
        m = frame.m3_total_mw.to_numpy(float)
        e = frame.residual_total_mw.to_numpy(float)
        result.update({"m3_std_mw": float(m.std()), "residual_std_mw": float(e.std()),
                       "mean_residual_mw": float(e.mean()),
                       "m3_residual_covariance_mw2": float(np.mean((m-m.mean())*(e-e.mean())))})
    return result
