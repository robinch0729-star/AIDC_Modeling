"""Aggregate M3+R2 and M3+R3 NLR IT workload groups without a rho formula.

The NLR trace is treated as an equivalent workload-group profile, not as a
verified physical server trace.  Facility-scale scenarios are formed by direct
capacity-weighted summation of group profiles:

    P_IT(t) = scale * sum_g w_g [P_M3(r_g, d_g) + e_g(t)]

Three explicit synchronization cases are compared: independent groups, a
partially synchronized subset, and a fully synchronized stress case.  These are
sensitivity scenarios because current NLR files do not identify facility-wide
cross-group correlation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .residual_model_selection import (
    ResidualParameterLibrary,
    build_parameter_library,
    simulate_residual,
)
from .residual_models import (
    fit_m3_by_dataset,
    load_condition_targets,
    load_profiles,
)


DEFAULT_TARGET_IT_MW = 20.0
DEFAULT_DURATION_S = 180.0
DEFAULT_INTERVAL_S = 0.1
DEFAULT_REQUEST_RATES = (10.0, 50.0, 90.0)
DEFAULT_GROUP_COUNTS = (1, 10, 50)
DEFAULT_MODELS = ("R2_ou", "R3_jump_ou")
SYNCHRONIZATION_SCENARIOS = {
    "independent": 0.0,
    "partial_25pct": 0.25,
    "fully_synchronized": 1.0,
}


@dataclass
class WorkloadAggregationGenerator:
    m3_parameters: Mapping[str, Any]
    residual_library: ResidualParameterLibrary
    calibration_splits: tuple[str, ...]

    @property
    def prompt_datasets(self) -> tuple[str, ...]:
        return tuple(sorted(self.m3_parameters))

    def generate_unit_profile(
        self,
        *,
        residual_model: str,
        prompt_dataset: str,
        request_rate: float,
        duration_s: float = DEFAULT_DURATION_S,
        interval_s: float = DEFAULT_INTERVAL_S,
        seed: int = 42,
    ) -> pd.DataFrame:
        if residual_model not in DEFAULT_MODELS:
            raise KeyError(residual_model)
        if prompt_dataset not in self.m3_parameters:
            raise KeyError(prompt_dataset)
        length = max(30, int(round(duration_s / interval_s)))
        steady_kw = float(self.m3_parameters[prompt_dataset].predict(request_rate))
        residual = simulate_residual(
            residual_model,
            length=length,
            prompt_dataset=prompt_dataset,
            request_rate=request_rate,
            library=self.residual_library,
            rng=np.random.default_rng(seed),
        )
        power = np.maximum(steady_kw + residual, 0.0)
        return pd.DataFrame(
            {
                "time_s": np.arange(length, dtype=float) * interval_s,
                "residual_model": residual_model,
                "prompt_dataset": prompt_dataset,
                "request_rate": request_rate,
                "steady_power_kw": steady_kw,
                "residual_kw": power - steady_kw,
                "power_kw": power,
            }
        )

    def generate_aggregate_profile(
        self,
        *,
        residual_model: str,
        prompt_dataset: str,
        request_rate: float,
        target_it_mean_mw: float,
        workload_group_count: int,
        synchronization_scenario: str,
        duration_s: float = DEFAULT_DURATION_S,
        interval_s: float = DEFAULT_INTERVAL_S,
        seed: int = 42,
    ) -> pd.DataFrame:
        if residual_model not in DEFAULT_MODELS:
            raise KeyError(residual_model)
        if prompt_dataset not in self.m3_parameters:
            raise KeyError(prompt_dataset)
        if synchronization_scenario not in SYNCHRONIZATION_SCENARIOS:
            raise KeyError(synchronization_scenario)
        if target_it_mean_mw <= 0:
            raise ValueError("target_it_mean_mw must be positive")
        if workload_group_count < 1:
            raise ValueError("workload_group_count must be at least one")

        length = max(30, int(round(duration_s / interval_s)))
        steady_kw = float(self.m3_parameters[prompt_dataset].predict(request_rate))
        requested_fraction = SYNCHRONIZATION_SCENARIOS[synchronization_scenario]
        if synchronization_scenario == "independent":
            synchronized_groups = 0
        elif synchronization_scenario == "fully_synchronized":
            synchronized_groups = workload_group_count
        else:
            synchronized_groups = max(1, int(round(requested_fraction * workload_group_count)))
            synchronized_groups = min(synchronized_groups, workload_group_count)
        independent_groups = workload_group_count - synchronized_groups
        rng = np.random.default_rng(seed)
        residual_sum = np.zeros(length, dtype=float)

        if synchronized_groups:
            common = simulate_residual(
                residual_model,
                length=length,
                prompt_dataset=prompt_dataset,
                request_rate=request_rate,
                library=self.residual_library,
                rng=np.random.default_rng(int(rng.integers(0, 2**32 - 1))),
            )
            residual_sum += synchronized_groups * common

        for _ in range(independent_groups):
            residual_sum += simulate_residual(
                residual_model,
                length=length,
                prompt_dataset=prompt_dataset,
                request_rate=request_rate,
                library=self.residual_library,
                rng=np.random.default_rng(int(rng.integers(0, 2**32 - 1))),
            )

        group_mean_residual_kw = residual_sum / workload_group_count
        equivalent_power_kw = np.maximum(steady_kw + group_mean_residual_kw, 0.0)
        scale = target_it_mean_mw / float(np.mean(equivalent_power_kw))
        it_power_mw = equivalent_power_kw * scale
        return pd.DataFrame(
            {
                "time_s": np.arange(length, dtype=float) * interval_s,
                "residual_model": residual_model,
                "prompt_dataset": prompt_dataset,
                "request_rate": request_rate,
                "target_it_mean_mw": target_it_mean_mw,
                "workload_group_count": workload_group_count,
                "synchronization_scenario": synchronization_scenario,
                "requested_synchronized_fraction": requested_fraction,
                "synchronized_group_count": synchronized_groups,
                "independent_group_count": independent_groups,
                "realized_synchronized_fraction": synchronized_groups / workload_group_count,
                "equivalent_unit_steady_kw": steady_kw,
                "equivalent_group_mean_power_kw": equivalent_power_kw,
                "it_power_mw": it_power_mw,
            }
        )


def build_generator_from_nlr(
    *,
    dataset_root: str | Path,
    manifest_path: str | Path,
    condition_target_path: str | Path,
    split_path: str | Path,
    calibration_splits: Sequence[str] = ("train", "validation"),
) -> WorkloadAggregationGenerator:
    profiles = load_profiles(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split_path=split_path,
    )
    targets = load_condition_targets(condition_target_path, split_path)
    allowed = set(calibration_splits)
    m3_parameters = fit_m3_by_dataset(targets, allowed)
    residual_library = build_parameter_library(profiles, m3_parameters, allowed)
    return WorkloadAggregationGenerator(
        m3_parameters=m3_parameters,
        residual_library=residual_library,
        calibration_splits=tuple(calibration_splits),
    )


def profile_metrics(profile: pd.DataFrame) -> dict[str, float]:
    power = pd.to_numeric(profile["it_power_mw"], errors="coerce").to_numpy(float)
    time_s = pd.to_numeric(profile["time_s"], errors="coerce").to_numpy(float)
    valid = np.isfinite(power) & np.isfinite(time_s)
    power, time_s = power[valid], time_s[valid]
    if len(power) < 30:
        raise ValueError("Aggregate profile has fewer than 30 samples")
    ramp = np.diff(power) / np.diff(time_s)
    mean = float(np.mean(power))
    std = float(np.std(power, ddof=0))

    def acf(lag_s: float) -> float:
        interval = float(np.median(np.diff(time_s)))
        lag = max(1, int(round(lag_s / interval)))
        centered = power - mean
        denominator = float(np.dot(centered, centered))
        if len(centered) <= lag + 2 or denominator <= np.finfo(float).eps:
            return 0.0
        return float(np.dot(centered[:-lag], centered[lag:]) / denominator)

    return {
        "mean_it_power_mw": mean,
        "std_it_power_mw": std,
        "coefficient_of_variation": std / max(mean, 1e-12),
        "minimum_it_power_mw": float(np.min(power)),
        "maximum_it_power_mw": float(np.max(power)),
        "p01_it_power_mw": float(np.quantile(power, 0.01)),
        "p05_it_power_mw": float(np.quantile(power, 0.05)),
        "p95_it_power_mw": float(np.quantile(power, 0.95)),
        "p99_it_power_mw": float(np.quantile(power, 0.99)),
        "peak_to_average_ratio": float(np.max(power) / max(mean, 1e-12)),
        "p95_abs_ramp_mw_per_s": float(np.quantile(np.abs(ramp), 0.95)),
        "p99_abs_ramp_mw_per_s": float(np.quantile(np.abs(ramp), 0.99)),
        "maximum_ramp_up_mw_per_s": float(np.max(ramp)),
        "maximum_ramp_down_mw_per_s": float(np.min(ramp)),
        "acf_1s": acf(1.0),
        "acf_10s": acf(10.0),
        "upper_flex_requirement_mw": float(np.max(power) - mean),
        "lower_flex_requirement_mw": float(mean - np.min(power)),
        "p99_upper_envelope_mw": float(np.quantile(power, 0.99)),
        "p01_lower_envelope_mw": float(np.quantile(power, 0.01)),
    }


def _stable_seed(base_seed: int, *parts: str) -> int:
    return int((base_seed + zlib.crc32("|".join(parts).encode("utf-8"))) % (2**32 - 1))


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [
        column if isinstance(column, str) else "_".join(str(item) for item in column if item)
        for column in result.columns.to_flat_index()
    ]
    return result


def run_aggregation_case_study(
    *,
    generator: WorkloadAggregationGenerator,
    output_dir: str | Path,
    target_it_mean_mw: float = DEFAULT_TARGET_IT_MW,
    residual_models: Sequence[str] = DEFAULT_MODELS,
    request_rates: Sequence[float] = DEFAULT_REQUEST_RATES,
    workload_group_counts: Sequence[int] = DEFAULT_GROUP_COUNTS,
    synchronization_scenarios: Sequence[str] = tuple(SYNCHRONIZATION_SCENARIOS),
    draws: int = 10,
    duration_s: float = DEFAULT_DURATION_S,
    interval_s: float = DEFAULT_INTERVAL_S,
    seed: int = 42,
) -> dict[str, Any]:
    if draws < 2:
        raise ValueError("draws must be at least two")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)

    m3_rows: list[dict[str, Any]] = []
    for dataset, parameters in generator.m3_parameters.items():
        m3_rows.extend(
            {
                "prompt_dataset": dataset,
                "parameter": name,
                "value": float(value),
            }
            for name, value in (
                ("p0_kw", parameters.p0_kw),
                ("delta_p_kw", parameters.delta_p_kw),
                ("k", parameters.k),
                ("x0", parameters.x0),
                ("saturation_power_kw", parameters.saturation_power_kw),
                ("half_rate_request_s", parameters.half_rate_request_s),
            )
        )
    pd.DataFrame(m3_rows).to_csv(output / "m3_parameters.csv", index=False)
    generator.residual_library.parameters.to_csv(
        output / "residual_parameter_library.csv", index=False
    )

    scenarios = [
        (model, dataset, float(rate), int(groups), sync, draw)
        for model in residual_models
        for dataset in generator.prompt_datasets
        for rate in request_rates
        for groups in workload_group_counts
        for sync in synchronization_scenarios
        for draw in range(draws)
    ]
    rows: list[dict[str, Any]] = []
    representative: list[pd.DataFrame] = []
    for model, dataset, rate, groups, sync, draw in tqdm(
        scenarios, desc="R2/R3 direct workload-group aggregation"
    ):
        scenario_seed = _stable_seed(
            seed, model, dataset, str(rate), str(groups), sync, str(draw)
        )
        profile = generator.generate_aggregate_profile(
            residual_model=model,
            prompt_dataset=dataset,
            request_rate=rate,
            target_it_mean_mw=target_it_mean_mw,
            workload_group_count=groups,
            synchronization_scenario=sync,
            duration_s=duration_s,
            interval_s=interval_s,
            seed=scenario_seed,
        )
        rows.append(
            {
                "residual_model": model,
                "prompt_dataset": dataset,
                "request_rate": rate,
                "target_it_mean_mw": target_it_mean_mw,
                "workload_group_count": groups,
                "synchronization_scenario": sync,
                "requested_synchronized_fraction": float(
                    profile["requested_synchronized_fraction"].iloc[0]
                ),
                "realized_synchronized_fraction": float(
                    profile["realized_synchronized_fraction"].iloc[0]
                ),
                "draw": draw,
                **profile_metrics(profile),
            }
        )
        if rate == max(request_rates) and groups == max(workload_group_counts) and draw == 0:
            representative.append(profile)

    scenario_metrics = pd.DataFrame(rows)
    group_columns = [
        "residual_model",
        "prompt_dataset",
        "request_rate",
        "target_it_mean_mw",
        "workload_group_count",
        "synchronization_scenario",
        "requested_synchronized_fraction",
        "realized_synchronized_fraction",
    ]
    metric_columns = [
        "std_it_power_mw",
        "coefficient_of_variation",
        "minimum_it_power_mw",
        "maximum_it_power_mw",
        "p01_it_power_mw",
        "p99_it_power_mw",
        "p95_abs_ramp_mw_per_s",
        "p99_abs_ramp_mw_per_s",
        "maximum_ramp_up_mw_per_s",
        "maximum_ramp_down_mw_per_s",
        "acf_1s",
        "acf_10s",
        "upper_flex_requirement_mw",
        "lower_flex_requirement_mw",
    ]
    summary = _flatten_columns(
        scenario_metrics.groupby(group_columns, observed=True)[metric_columns]
        .agg(["mean", "std", "median"])
        .reset_index()
    )

    comparison_index = [
        "prompt_dataset",
        "request_rate",
        "workload_group_count",
        "synchronization_scenario",
    ]
    compare_metrics = [
        "std_it_power_mw_mean",
        "p01_it_power_mw_mean",
        "p99_it_power_mw_mean",
        "p95_abs_ramp_mw_per_s_mean",
        "p99_abs_ramp_mw_per_s_mean",
        "upper_flex_requirement_mw_mean",
        "lower_flex_requirement_mw_mean",
    ]
    wide = summary.pivot_table(
        index=comparison_index,
        columns="residual_model",
        values=compare_metrics,
        aggfunc="first",
    )
    comparison_rows: list[dict[str, Any]] = []
    for index, row in wide.iterrows():
        item = dict(zip(comparison_index, index, strict=True))
        for metric in compare_metrics:
            r2 = float(row[(metric, "R2_ou")])
            r3 = float(row[(metric, "R3_jump_ou")])
            item[f"R2_{metric}"] = r2
            item[f"R3_{metric}"] = r3
            item[f"R3_minus_R2_{metric}"] = r3 - r2
        comparison_rows.append(item)
    model_comparison = pd.DataFrame(comparison_rows)

    representative_table = (
        pd.concat(representative, ignore_index=True) if representative else pd.DataFrame()
    )
    dso_information = summary[
        [
            *group_columns,
            "p01_it_power_mw_mean",
            "p99_it_power_mw_mean",
            "p95_abs_ramp_mw_per_s_mean",
            "p99_abs_ramp_mw_per_s_mean",
            "upper_flex_requirement_mw_mean",
            "lower_flex_requirement_mw_mean",
            "acf_1s_mean",
            "acf_10s_mean",
        ]
    ].rename(
        columns={
            "p01_it_power_mw_mean": "estimated_import_p01_mw",
            "p99_it_power_mw_mean": "estimated_import_p99_mw",
            "p95_abs_ramp_mw_per_s_mean": "estimated_abs_ramp_p95_mw_per_s",
            "p99_abs_ramp_mw_per_s_mean": "estimated_abs_ramp_p99_mw_per_s",
            "upper_flex_requirement_mw_mean": "estimated_upper_flex_requirement_mw",
            "lower_flex_requirement_mw_mean": "estimated_lower_flex_requirement_mw",
            "acf_1s_mean": "estimated_acf_1s",
            "acf_10s_mean": "estimated_acf_10s",
        }
    )

    scenario_metrics.to_csv(output / "aggregation_scenario_metrics.csv", index=False)
    scenario_metrics.to_parquet(output / "aggregation_scenario_metrics.parquet", index=False)
    summary.to_csv(output / "aggregation_summary.csv", index=False)
    summary.to_parquet(output / "aggregation_summary.parquet", index=False)
    model_comparison.to_csv(output / "r3_minus_r2_aggregation_comparison.csv", index=False)
    dso_information.to_csv(output / "candidate_dso_information_table.csv", index=False)
    representative_table.to_csv(output / "representative_20mw_profiles.csv", index=False)
    if not representative_table.empty:
        representative_table.to_parquet(output / "representative_20mw_profiles.parquet", index=False)

    _plot_representative_profiles(
        representative_table,
        max(request_rates),
        max(workload_group_counts),
        figures / "representative_r2_r3_20mw_profiles.png",
    )
    _plot_uncertainty_envelopes(
        summary,
        max(request_rates),
        max(workload_group_counts),
        figures / "r2_r3_uncertainty_envelopes.png",
    )
    _plot_ramp_comparison(
        summary,
        max(request_rates),
        figures / "r2_r3_ramp_by_group_count.png",
    )

    decision = {
        "aggregation_version": 2,
        "residual_models": list(residual_models),
        "target_it_mean_mw": target_it_mean_mw,
        "target_interpretation": "mean IT power only; not PCC total facility demand",
        "aggregation_equation": "capacity-weighted direct sum of workload-group profiles",
        "workload_group_count_interpretation": (
            "equivalent workload-group scenario parameter; not a physical server/GPU count"
        ),
        "synchronization_scenarios": dict(SYNCHRONIZATION_SCENARIOS),
        "synchronization_interpretation": (
            "explicit sensitivity cases; not identified from current NLR traces"
        ),
        "old_rho_formula_used": False,
        "mad_used": False,
        "r2_r3_status": (
            "parallel candidates; final family selection deferred to DOE/decision-regret evaluation"
        ),
        "scenario_count": int(len(scenario_metrics)),
        "draws_per_case": draws,
        "calibration_splits": list(generator.calibration_splits),
    }
    (output / "aggregation_methodology_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = _aggregation_report(
        summary=summary,
        decision=decision,
        max_rate=max(request_rates),
        max_groups=max(workload_group_counts),
    )
    (output / "workload_aggregation_v2_report_ko.md").write_text(report, encoding="utf-8")
    return {
        "output_dir": output,
        "scenario_metrics": scenario_metrics,
        "summary": summary,
        "model_comparison": model_comparison,
        "dso_information": dso_information,
        "decision": decision,
    }


def _plot_representative_profiles(
    table: pd.DataFrame, rate: float, groups: int, output: Path
) -> None:
    if table.empty:
        return
    datasets = table["prompt_dataset"].drop_duplicates().tolist()
    models = list(DEFAULT_MODELS)
    fig, axes = plt.subplots(
        len(datasets), len(models), figsize=(14, 4.0 * len(datasets)), dpi=160, sharex=True
    )
    axes = np.asarray(axes).reshape(len(datasets), len(models))
    colors = {
        "independent": "#2563eb",
        "partial_25pct": "#f97316",
        "fully_synchronized": "#dc2626",
    }
    for row_index, dataset in enumerate(datasets):
        for column_index, model in enumerate(models):
            axis = axes[row_index, column_index]
            part = table.loc[
                table["prompt_dataset"].eq(dataset)
                & table["residual_model"].eq(model)
            ]
            for scenario, group in part.groupby("synchronization_scenario", observed=True):
                axis.plot(
                    group["time_s"],
                    group["it_power_mw"],
                    lw=0.8,
                    color=colors[str(scenario)],
                    label=str(scenario),
                )
            axis.axhline(20.0, color="#111827", lw=0.8, alpha=0.6)
            axis.set_title(f"{dataset} | {model}")
            axis.set_ylabel("IT power (MW)")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    fig.suptitle(f"{rate:g} request/s | {groups} equivalent workload groups")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_uncertainty_envelopes(
    summary: pd.DataFrame, rate: float, groups: int, output: Path
) -> None:
    selected = summary.loc[
        summary["request_rate"].eq(rate)
        & summary["workload_group_count"].eq(groups)
    ].copy()
    datasets = selected["prompt_dataset"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), dpi=160, sharey=True)
    axes = np.atleast_1d(axes)
    for axis, dataset in zip(axes, datasets, strict=True):
        part = selected.loc[selected["prompt_dataset"].eq(dataset)].copy()
        labels = list(SYNCHRONIZATION_SCENARIOS)
        x = np.arange(len(labels), dtype=float)
        width = 0.34
        for offset, model, color in (
            (-width / 2, "R2_ou", "#2563eb"),
            (width / 2, "R3_jump_ou", "#f97316"),
        ):
            model_part = part.loc[part["residual_model"].eq(model)].set_index(
                "synchronization_scenario"
            )
            lower = np.array([model_part.loc[label, "p01_it_power_mw_mean"] for label in labels])
            upper = np.array([model_part.loc[label, "p99_it_power_mw_mean"] for label in labels])
            axis.bar(
                x + offset,
                upper - lower,
                width,
                bottom=lower,
                color=color,
                alpha=0.75,
                label=model,
            )
        axis.axhline(20.0, color="#111827", lw=0.8)
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.set_title(str(dataset))
        axis.set_ylabel("P01--P99 IT import envelope (MW)")
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    fig.suptitle(f"Candidate DSO information | {rate:g} request/s | {groups} groups")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_ramp_comparison(summary: pd.DataFrame, rate: float, output: Path) -> None:
    selected = summary.loc[summary["request_rate"].eq(rate)].copy()
    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    styles = {"R2_ou": "-", "R3_jump_ou": "--"}
    markers = {
        "independent": "o",
        "partial_25pct": "s",
        "fully_synchronized": "^",
    }
    for (dataset, model, sync), group in selected.groupby(
        ["prompt_dataset", "residual_model", "synchronization_scenario"], observed=True
    ):
        ordered = group.sort_values("workload_group_count")
        ax.plot(
            ordered["workload_group_count"],
            ordered["p99_abs_ramp_mw_per_s_mean"],
            linestyle=styles[str(model)],
            marker=markers[str(sync)],
            label=f"{dataset} | {model} | {sync}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Equivalent workload-group count")
    ax.set_ylabel("P99 absolute IT ramp (MW/s)")
    ax.set_title(f"R2/R3 aggregation ramp comparison at {rate:g} request/s")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [
            f"{float(value):.4g}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _aggregation_report(
    *,
    summary: pd.DataFrame,
    decision: Mapping[str, Any],
    max_rate: float,
    max_groups: int,
) -> str:
    selected = summary.loc[
        summary["request_rate"].eq(max_rate)
        & summary["workload_group_count"].eq(max_groups)
    ][
        [
            "residual_model",
            "prompt_dataset",
            "synchronization_scenario",
            "realized_synchronized_fraction",
            "std_it_power_mw_mean",
            "p01_it_power_mw_mean",
            "p99_it_power_mw_mean",
            "p99_abs_ramp_mw_per_s_mean",
            "upper_flex_requirement_mw_mean",
            "lower_flex_requirement_mw_mean",
        ]
    ].round(5)
    return rf"""# R2/R3 병렬 20 MW IT workload aggregation v2

## 변경사항

- 기존 MAD 기반 R3와 `sqrt(rho)` common-factor 식을 사용하지 않음
- M3+R2 OU와 M3+condition-aware Jump-OU를 병렬 유지
- workload group profile을 동일 용량가중치로 직접 합산
- independent, partial 25%, fully synchronized stress case를 명시적으로 구성

## 집합화 식

$$
P_{{IT}}(t)=s\sum_{{g=1}}^G w_g
\left[P_{{M3}}(r_g,d_g)+e_g(t)\right],
\qquad \sum_g w_g=1
$$

- $s$: 평균 IT 전력을 {decision['target_it_mean_mw']:g} MW로 맞추는 scale
- $G$: equivalent workload-group 수이며 실제 서버/GPU 수가 아님
- partial synchronization은 일부 group에 동일 residual realization을 배정하여 구현

## {max_rate:g} request/s, {max_groups} group 대표 결과

{_markdown_table(selected.reset_index(drop=True))}

## DSO 전달정보 후보

- 예상 평균 IT import
- P01--P99 import envelope
- P95/P99 absolute ramp
- 상·하향 유연성 필요량
- 1 s 및 10 s ACF

## 해석 경계

- 현재 20 MW는 IT 부하만 의미하며 냉각·UPS loss·보조부하가 포함된 PCC 전력이 아님.
- NLR 파일의 물리적 server/GPU 대표단위가 확정되지 않았으므로 group count는 sensitivity parameter임.
- synchronization 비율은 현재 NLR 데이터에서 식별된 값이 아니라 독립·부분동기·완전동기 stress scenario임.
- R2/R3 최종 family는 본 단계에서 선택하지 않고 DOE 위반과 decision regret 비교까지 유보함.
"""
