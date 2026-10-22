"""Preliminary heterogeneous aggregation of M3+R3 equivalent IT-load units.

This module replaces the legacy operation of multiplying one stochastic trace
to a MW target.  It samples prompt-dataset/request-rate conditions for many
equivalent workload units, generates an independent R3 realization per unit,
and directly sums their power.

Scientific boundary
-------------------
The NLR measurement unit has not been established as one physical server or
GPU node.  Consequently, ``equivalent_unit_count`` must not be reported as a
physical node count.  The workload shares and uniform request-rate laws are
explicit preliminary assumptions, not distributions estimated from NLR data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .final_it_profile_generator import (
    CALIBRATED_INTERVAL_S,
    FIXED_RESIDUAL_MODEL,
    FinalITProfileGenerator,
)
from .residual_model_selection import simulate_residual_ensemble


@dataclass(frozen=True)
class WorkloadSpec:
    key: str
    label: str
    prompt_dataset: str
    share: float
    request_rate_low: int
    request_rate_high: int

    def validate(self) -> None:
        if not self.key or not self.label:
            raise ValueError("Workload key and label must be non-empty")
        if not np.isfinite(self.share) or self.share <= 0:
            raise ValueError("Workload share must be finite and positive")
        if self.request_rate_low < 0 or self.request_rate_high < self.request_rate_low:
            raise ValueError("Invalid request-rate range")


@dataclass(frozen=True)
class FacilityAggregationRequest:
    target_pcc_mean_mw: float = 20.0
    pue: float = 1.15
    duration_s: float = 180.0
    interval_s: float = CALIBRATED_INTERVAL_S
    facility_scenarios: int = 30
    master_seed: int = 20260909

    @property
    def target_it_mean_mw(self) -> float:
        return self.target_pcc_mean_mw / self.pue

    def validate(self) -> None:
        if not np.isfinite(self.target_pcc_mean_mw) or self.target_pcc_mean_mw <= 0:
            raise ValueError("target_pcc_mean_mw must be finite and positive")
        if not np.isfinite(self.pue) or self.pue < 1:
            raise ValueError("pue must be finite and at least one")
        if not np.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("duration_s must be finite and positive")
        if not np.isclose(self.interval_s, CALIBRATED_INTERVAL_S):
            raise ValueError(
                f"R3 parameters are calibrated at dt={CALIBRATED_INTERVAL_S:g} s"
            )
        if self.facility_scenarios < 1:
            raise ValueError("facility_scenarios must be at least one")


def allocate_unit_counts(total: int, workloads: Sequence[WorkloadSpec]) -> dict[str, int]:
    """Allocate an integer total with the largest-remainder method."""
    if total < 1:
        raise ValueError("total must be positive")
    if not workloads:
        raise ValueError("At least one workload is required")
    for workload in workloads:
        workload.validate()
    shares = np.asarray([workload.share for workload in workloads], dtype=float)
    shares = shares / shares.sum()
    raw = shares * total
    counts = np.floor(raw).astype(int)
    remainder = total - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[:remainder]] += 1
    return {workload.key: int(count) for workload, count in zip(workloads, counts)}


def expected_unit_mean_kw(
    generator: FinalITProfileGenerator, workload: WorkloadSpec
) -> float:
    """Expected M3 power under a discrete-uniform integer request-rate law."""
    workload.validate()
    if workload.prompt_dataset not in generator.m3_parameters:
        raise KeyError(f"Unknown prompt dataset: {workload.prompt_dataset}")
    rates = np.arange(
        workload.request_rate_low, workload.request_rate_high + 1, dtype=float
    )
    statuses = {
        generator.request_rate_status(workload.prompt_dataset, float(rate))
        for rate in (rates[0], rates[-1])
    }
    if "unsupported_extrapolation" in statuses:
        raise ValueError(
            f"{workload.label} request-rate range includes unsupported extrapolation"
        )
    return float(np.mean(generator.m3_parameters[workload.prompt_dataset].predict(rates)))


def choose_equivalent_unit_count(
    generator: FinalITProfileGenerator,
    workloads: Sequence[WorkloadSpec],
    target_it_mean_mw: float,
    *,
    search_radius: int = 100,
) -> tuple[int, dict[str, int], float]:
    """Choose the integer unit count whose expected M3 sum is nearest the target."""
    if not np.isfinite(target_it_mean_mw) or target_it_mean_mw <= 0:
        raise ValueError("target_it_mean_mw must be finite and positive")
    means = {
        workload.key: expected_unit_mean_kw(generator, workload)
        for workload in workloads
    }
    normalized_shares = np.asarray([workload.share for workload in workloads], float)
    normalized_shares /= normalized_shares.sum()
    expected_single = float(
        sum(
            share * means[workload.key]
            for share, workload in zip(normalized_shares, workloads)
        )
    )
    center = max(1, int(round(target_it_mean_mw * 1000.0 / expected_single)))
    best: tuple[float, int, dict[str, int], float] | None = None
    for total in range(max(1, center - search_radius), center + search_radius + 1):
        counts = allocate_unit_counts(total, workloads)
        expected_kw = float(sum(counts[key] * means[key] for key in counts))
        candidate = (abs(expected_kw - target_it_mean_mw * 1000.0), total, counts, expected_kw)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return best[1], best[2], best[3] / 1000.0


def _simulate_component_kw(
    generator: FinalITProfileGenerator,
    workload: WorkloadSpec,
    unit_count: int,
    length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """Generate and sum independent unit traces without retaining unit-level data."""
    rates = rng.integers(
        workload.request_rate_low,
        workload.request_rate_high + 1,
        size=unit_count,
    )
    component_kw = np.zeros(length, dtype=float)
    minimum_unit_power_kw = np.inf
    maximum_unit_power_kw = -np.inf
    for rate in np.unique(rates):
        draws = int(np.count_nonzero(rates == rate))
        steady_kw = float(
            generator.m3_parameters[workload.prompt_dataset].predict(float(rate))
        )
        residual_kw = simulate_residual_ensemble(
            FIXED_RESIDUAL_MODEL,
            draws=draws,
            length=length,
            prompt_dataset=workload.prompt_dataset,
            request_rate=float(rate),
            library=generator.residual_library,
            rng=rng,
        )
        unit_power_kw = steady_kw + residual_kw
        minimum_unit_power_kw = min(minimum_unit_power_kw, float(unit_power_kw.min()))
        maximum_unit_power_kw = max(maximum_unit_power_kw, float(unit_power_kw.max()))
        component_kw += unit_power_kw.sum(axis=0)
    if minimum_unit_power_kw < 0:
        raise RuntimeError("At least one generated equivalent-unit power became negative")
    return component_kw, {
        "unit_count": int(unit_count),
        "mean_request_rate": float(np.mean(rates)),
        "minimum_request_rate": int(np.min(rates)),
        "maximum_request_rate": int(np.max(rates)),
        "minimum_unit_power_kw": minimum_unit_power_kw,
        "maximum_unit_power_kw": maximum_unit_power_kw,
    }


def _profile_metrics(values_mw: np.ndarray, interval_s: float) -> dict[str, float]:
    power = np.asarray(values_mw, dtype=float)
    ramp_0p1 = np.diff(power) / interval_s
    lag_1s = int(round(1.0 / interval_s))
    ramp_1s = (power[lag_1s:] - power[:-lag_1s]) / 1.0
    return {
        "mean_it_mw": float(np.mean(power)),
        "std_it_mw": float(np.std(power, ddof=0)),
        "minimum_it_mw": float(np.min(power)),
        "maximum_it_mw": float(np.max(power)),
        "p01_it_mw": float(np.quantile(power, 0.01)),
        "p99_it_mw": float(np.quantile(power, 0.99)),
        "p95_abs_ramp_0p1s_mw_per_s": float(np.quantile(np.abs(ramp_0p1), 0.95)),
        "maximum_ramp_up_0p1s_mw_per_s": float(np.max(ramp_0p1)),
        "maximum_ramp_down_0p1s_mw_per_s": float(np.min(ramp_0p1)),
        "p95_abs_ramp_1s_mw_per_s": float(np.quantile(np.abs(ramp_1s), 0.95)),
        "maximum_ramp_up_1s_mw_per_s": float(np.max(ramp_1s)),
        "maximum_ramp_down_1s_mw_per_s": float(np.min(ramp_1s)),
    }


def run_heterogeneous_aggregation(
    generator: FinalITProfileGenerator,
    workloads: Sequence[WorkloadSpec],
    request: FacilityAggregationRequest,
    output_dir: str | Path,
) -> Path:
    """Run the preliminary heterogeneous facility experiment and save artifacts."""
    request.validate()
    if not workloads:
        raise ValueError("At least one workload is required")
    if not np.isclose(sum(workload.share for workload in workloads), 1.0, atol=1e-10):
        raise ValueError("Workload shares must sum to one")
    if len({workload.key for workload in workloads}) != len(workloads):
        raise ValueError("Workload keys must be unique")
    for workload in workloads:
        workload.validate()

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    length = int(round(request.duration_s / request.interval_s))
    if length < 2 or not np.isclose(length * request.interval_s, request.duration_s):
        raise ValueError("duration_s must be an integer multiple of interval_s")

    total_units, unit_counts, expected_mean_mw = choose_equivalent_unit_count(
        generator,
        workloads,
        request.target_it_mean_mw,
    )
    time_s = np.arange(length, dtype=float) * request.interval_s
    profile_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int]] = []
    condition_rows: list[dict[str, float | int | str]] = []
    started = time.perf_counter()

    for scenario_id in range(request.facility_scenarios):
        seed_sequence = np.random.SeedSequence([request.master_seed, scenario_id])
        component_sequences = seed_sequence.spawn(len(workloads))
        components_mw: dict[str, np.ndarray] = {}
        for workload, sequence in zip(workloads, component_sequences):
            component_kw, condition = _simulate_component_kw(
                generator,
                workload,
                unit_counts[workload.key],
                length,
                np.random.default_rng(sequence),
            )
            components_mw[workload.key] = component_kw / 1000.0
            condition_rows.append(
                {
                    "scenario_id": scenario_id,
                    "workload_key": workload.key,
                    "prompt_dataset": workload.prompt_dataset,
                    **condition,
                }
            )
        total_mw = np.sum(np.stack(list(components_mw.values())), axis=0)
        frame = pd.DataFrame(
            {
                "scenario_id": scenario_id,
                "sample_index": np.arange(length, dtype=int),
                "time_s": time_s,
                "total_it_mw": total_mw,
                **{f"{key}_it_mw": value for key, value in components_mw.items()},
            }
        )
        profile_rows.append(frame)
        metrics = _profile_metrics(total_mw, request.interval_s)
        summary_rows.append(
            {
                "scenario_id": scenario_id,
                **metrics,
                "target_it_mean_mw": request.target_it_mean_mw,
                "mean_error_mw": metrics["mean_it_mw"] - request.target_it_mean_mw,
                "mean_error_pct": 100.0
                * (metrics["mean_it_mw"] - request.target_it_mean_mw)
                / request.target_it_mean_mw,
            }
        )

    profiles = pd.concat(profile_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    conditions = pd.DataFrame(condition_rows)
    runtime_s = time.perf_counter() - started
    profiles.to_csv(output / "facility_profiles_long.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    conditions.to_csv(output / "sampled_conditions.csv", index=False, encoding="utf-8-sig")

    wide = profiles.pivot(index="time_s", columns="scenario_id", values="total_it_mw")
    wide.columns = [f"scenario_{int(column):03d}_total_it_mw" for column in wide.columns]
    wide.reset_index().to_csv(
        output / "facility_profiles_wide.csv", index=False, encoding="utf-8-sig"
    )

    configuration = {
        "experiment": "preliminary heterogeneous equivalent-workload aggregation",
        "request": asdict(request),
        "target_it_mean_mw": request.target_it_mean_mw,
        "workloads": [asdict(workload) for workload in workloads],
        "request_rate_distribution": "discrete uniform, integer request/s",
        "residual_model": FIXED_RESIDUAL_MODEL,
        "residual_dependence": "independent across equivalent units",
        "equivalent_unit_count": total_units,
        "unit_counts": unit_counts,
        "expected_m3_mean_mw": expected_mean_mw,
        "post_aggregation_scaling_applied": False,
        "runtime_s": runtime_s,
        "scientific_boundary": (
            "Equivalent workload units are not claimed as physical GPU/server nodes; "
            "workload shares and request-rate distributions are preliminary assumptions."
        ),
    }
    (output / "generation_config.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _plot_profiles(
        profiles,
        workloads,
        request.target_it_mean_mw,
        output / "heterogeneous_facility_it_profiles.png",
    )
    _plot_ramp_summary(summary, output / "facility_ramp_summary.png")
    (output / "report_ko.md").write_text(
        _report(configuration, summary, conditions), encoding="utf-8"
    )
    return output


def _plot_profiles(
    profiles: pd.DataFrame,
    workloads: Sequence[WorkloadSpec],
    target_it_mean_mw: float,
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), dpi=170, sharex=True)
    selected = sorted(profiles["scenario_id"].unique())[:5]
    for scenario_id in selected:
        part = profiles.loc[profiles["scenario_id"].eq(scenario_id)]
        axes[0].plot(
            part["time_s"],
            part["total_it_mw"],
            lw=0.9,
            alpha=0.85,
            label=f"facility scenario {int(scenario_id)}",
        )
    axes[0].axhline(
        target_it_mean_mw,
        color="black",
        ls="--",
        lw=1.2,
        label=f"target IT mean {target_it_mean_mw:.3f} MW",
    )
    axes[0].set_ylabel("Total IT power (MW)")
    axes[0].set_title("Heterogeneous equivalent-workload aggregation (independent baseline)")
    axes[0].legend(fontsize=8, ncol=3)
    axes[0].grid(alpha=0.25)

    example = profiles.loc[profiles["scenario_id"].eq(selected[0])]
    component_values = [example[f"{workload.key}_it_mw"].to_numpy() for workload in workloads]
    axes[1].stackplot(
        example["time_s"],
        component_values,
        labels=[f"{workload.label} ({100*workload.share:.0f}% units)" for workload in workloads],
        alpha=0.75,
    )
    axes[1].plot(
        example["time_s"],
        example["total_it_mw"],
        color="black",
        lw=0.8,
        label="total IT",
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("IT power contribution (MW)")
    axes[1].set_title("Scenario 0: workload contribution")
    axes[1].legend(fontsize=8, ncol=3, loc="upper right")
    axes[1].grid(alpha=0.25)
    figure.suptitle("30% InstructCoder + 70% Llama2-data | preliminary 17.39 MW IT profile")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_ramp_summary(summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=170)
    axes[0].boxplot(
        summary["p95_abs_ramp_0p1s_mw_per_s"],
        tick_labels=["0.1 s"],
        widths=0.4,
    )
    axes[0].set_ylabel("P95 absolute ramp (MW/s)")
    axes[0].set_title("Fast-ramp distribution across scenarios")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].boxplot(
        summary["p95_abs_ramp_1s_mw_per_s"],
        tick_labels=["1 s"],
        widths=0.4,
    )
    axes[1].set_ylabel("P95 absolute ramp (MW/s)")
    axes[1].set_title("One-second ramp distribution")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _report(config: dict, summary: pd.DataFrame, conditions: pd.DataFrame) -> str:
    request = config["request"]
    aggregate = summary.describe().loc[["mean", "std", "min", "max"]].round(6)
    condition_summary = (
        conditions.groupby(["workload_key", "prompt_dataset"], as_index=False)
        .agg(
            unit_count=("unit_count", "first"),
            mean_sampled_request_rate=("mean_request_rate", "mean"),
            minimum_sampled_request_rate=("minimum_request_rate", "min"),
            maximum_sampled_request_rate=("maximum_request_rate", "max"),
        )
        .round(4)
    )
    return f"""# Heterogeneous equivalent-workload IT aggregation preliminary result

## Assumptions

- Workload unit mix: 30% InstructCoder, 70% Llama2-data
- Request-rate law: discrete uniform (InstructCoder 10-100, Llama2-data 5-80 request/s)
- Residual dependence: independent R3 realization per equivalent unit
- Target PCC mean: {request['target_pcc_mean_mw']:.3f} MW
- Assumed PUE: {request['pue']:.3f}
- Target IT mean: {config['target_it_mean_mw']:.6f} MW
- Equivalent workload units: {config['equivalent_unit_count']:,}
- Post-aggregation scaling: not applied
- Runtime: {config['runtime_s']:.3f} s

## Sampled workload conditions

```text
{condition_summary.to_string(index=False)}
```

## Scenario metrics

```text
{aggregate.to_string()}
```

![Facility IT profiles](heterogeneous_facility_it_profiles.png)

![Facility ramp summary](facility_ramp_summary.png)

## Interpretation boundary

- This is a heterogeneous equivalent-workload aggregation, not a validated physical node aggregation.
- The 30/70 share and uniform request-rate distributions are preliminary scenario assumptions.
- Cooling, UPS losses, and auxiliary load are not included; the result is IT power, not PCC power.
- Fully independent residuals provide a diversity baseline and may underestimate common-mode facility ramps.
- Constant per-unit request rates are sampled at the beginning of each 180 s scenario; traffic-driven transitions are not modeled.
"""

