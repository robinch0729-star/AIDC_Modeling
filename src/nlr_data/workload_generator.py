"""Portable NLR M3+R3 workload generator and facility aggregation study.

The NLR online-inference traces are job-total IT power measurements, not a
documented one-server population sample.  Facility scaling therefore uses an
*equivalent statistical unit* rather than claiming a physical server count.
The target MW value fixes mean IT power; ``n_effective`` controls diversity and
``common_mode_fraction`` controls synchronization among equivalent units.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .dynamic_identifiability import M3Parameters
from .residual_models import (
    EventTemplate,
    PowerProfile,
    ResidualLibrary,
    build_residual_library,
    fit_m3_by_dataset,
    load_condition_targets,
    load_profiles,
    simulate_independent_residual_mean,
    simulate_residual,
)


DEFAULT_TARGET_IT_MW = 20.0
DEFAULT_DURATION_S = 180.0
DEFAULT_INTERVAL_S = 0.1
DEFAULT_EFFECTIVE_UNITS = (1, 10, 50, 200)
DEFAULT_COMMON_MODE_FRACTIONS = (0.0, 0.05, 0.25, 1.0)
DEFAULT_REQUEST_RATES = (10.0, 50.0, 90.0)


@dataclass
class WorkloadGenerator:
    m3_parameters: dict[str, M3Parameters]
    residual_library: ResidualLibrary
    residual_model: str
    artifact_metadata: dict[str, Any]

    @property
    def prompt_datasets(self) -> tuple[str, ...]:
        return tuple(sorted(self.m3_parameters))

    def generate_unit_profile(
        self,
        *,
        prompt_dataset: str,
        request_rate: float,
        duration_s: float = DEFAULT_DURATION_S,
        interval_s: float = DEFAULT_INTERVAL_S,
        seed: int = 42,
        residual_model: str | None = None,
    ) -> pd.DataFrame:
        if prompt_dataset not in self.m3_parameters:
            raise KeyError(
                f"Unknown prompt_dataset={prompt_dataset!r}; "
                f"available={self.prompt_datasets}"
            )
        if request_rate < 0 or duration_s <= 0 or interval_s <= 0:
            raise ValueError("request_rate must be nonnegative and time values positive")
        length = max(20, int(round(duration_s / interval_s)))
        model = residual_model or self.residual_model
        steady = float(self.m3_parameters[prompt_dataset].predict(request_rate))
        residual = simulate_residual(
            model,
            length=length,
            interval_s=interval_s,
            prompt_dataset=prompt_dataset,
            request_rate=request_rate,
            library=self.residual_library,
            rng=np.random.default_rng(seed),
        )
        power = np.maximum(steady + residual, 0.0)
        return pd.DataFrame(
            {
                "time_s": np.arange(length, dtype=float) * interval_s,
                "prompt_dataset": prompt_dataset,
                "request_rate": request_rate,
                "residual_model": model,
                "steady_power_kw": steady,
                "residual_kw": power - steady,
                "power_kw": power,
            }
        )

    def generate_facility_profile(
        self,
        *,
        prompt_dataset: str,
        request_rate: float,
        target_it_mean_mw: float,
        n_effective: int,
        common_mode_fraction: float,
        duration_s: float = DEFAULT_DURATION_S,
        interval_s: float = DEFAULT_INTERVAL_S,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate an aggregate IT-power profile with a common-factor model.

        For zero-mean unit residuals ``e_i`` and a common residual ``e_c``:

        e_agg = sqrt(rho) e_c + sqrt(1-rho) mean_i(e_i)

        If all terms have the same variance and are independent, the residual
        standard-deviation factor is sqrt(rho + (1-rho)/n_effective).
        ``n_effective`` is a diversity parameter, not an asserted server count.
        """
        if target_it_mean_mw <= 0:
            raise ValueError("target_it_mean_mw must be positive")
        if n_effective < 1:
            raise ValueError("n_effective must be at least one")
        if not 0.0 <= common_mode_fraction <= 1.0:
            raise ValueError("common_mode_fraction must be in [0, 1]")
        if prompt_dataset not in self.m3_parameters:
            raise KeyError(prompt_dataset)
        length = max(20, int(round(duration_s / interval_s)))
        steady_kw = float(self.m3_parameters[prompt_dataset].predict(request_rate))
        rng = np.random.default_rng(seed)

        common = np.zeros(length)
        if common_mode_fraction > 0.0:
            common = simulate_residual(
                self.residual_model,
                length=length,
                interval_s=interval_s,
                prompt_dataset=prompt_dataset,
                request_rate=request_rate,
                library=self.residual_library,
                rng=np.random.default_rng(int(rng.integers(0, 2**32 - 1))),
            )

        independent_mean = np.zeros(length)
        if common_mode_fraction < 1.0:
            independent_mean = simulate_independent_residual_mean(
                self.residual_model,
                n_units=n_effective,
                length=length,
                interval_s=interval_s,
                prompt_dataset=prompt_dataset,
                request_rate=request_rate,
                library=self.residual_library,
                rng=np.random.default_rng(int(rng.integers(0, 2**32 - 1))),
            )

        aggregate_residual_kw = (
            math.sqrt(common_mode_fraction) * common
            + math.sqrt(1.0 - common_mode_fraction) * independent_mean
        )
        equivalent_unit_power_kw = np.maximum(steady_kw + aggregate_residual_kw, 0.0)
        # Preserve the requested mean exactly.  The scale is statistical and
        # does not imply that one NLR trace corresponds to one physical server.
        scale_mw_per_kw = target_it_mean_mw / float(np.mean(equivalent_unit_power_kw))
        facility_power_mw = equivalent_unit_power_kw * scale_mw_per_kw
        theoretical_diversity = math.sqrt(
            common_mode_fraction
            + (1.0 - common_mode_fraction) / float(n_effective)
        )
        return pd.DataFrame(
            {
                "time_s": np.arange(length, dtype=float) * interval_s,
                "prompt_dataset": prompt_dataset,
                "request_rate": request_rate,
                "target_it_mean_mw": target_it_mean_mw,
                "n_effective": n_effective,
                "common_mode_fraction": common_mode_fraction,
                "theoretical_diversity_factor": theoretical_diversity,
                "equivalent_unit_steady_kw": steady_kw,
                "equivalent_unit_power_kw": equivalent_unit_power_kw,
                "it_power_mw": facility_power_mw,
            }
        )

    def save_artifact(self, output_dir: str | Path) -> Path:
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        stats = self.residual_library.statistics.copy()
        stats.to_parquet(output / "calibration_statistics.parquet", index=False)
        stats.to_csv(output / "calibration_statistics.csv", index=False, encoding="utf-8-sig")

        parameter_rows = []
        for dataset, parameter in self.m3_parameters.items():
            parameter_rows.append(
                {
                    "prompt_dataset": dataset,
                    "p0_kw": parameter.p0_kw,
                    "delta_p_kw": parameter.delta_p_kw,
                    "k": parameter.k,
                    "x0": parameter.x0,
                }
            )
        pd.DataFrame(parameter_rows).to_csv(
            output / "m3_parameters.csv", index=False, encoding="utf-8-sig"
        )

        residual_rows = []
        for profile_id, residual in self.residual_library.residuals.items():
            residual_rows.append(
                pd.DataFrame(
                    {
                        "profile_id": profile_id,
                        "sample_index": np.arange(len(residual)),
                        "residual_kw": residual,
                    }
                )
            )
        residual_table = pd.concat(residual_rows, ignore_index=True)
        residual_table.to_parquet(output / "residual_waveform_library.parquet", index=False)

        profile_rows = []
        for profile in self.residual_library.profiles.values():
            profile_rows.append(
                {
                    "profile_id": profile.profile_id,
                    "condition_id": profile.condition_id,
                    "prompt_dataset": profile.prompt_dataset,
                    "request_rate": profile.request_rate,
                    "split": profile.split,
                    "interval_s": profile.interval_s,
                    "measurement_origin": profile.measurement_origin,
                }
            )
        pd.DataFrame(profile_rows).to_csv(
            output / "profile_metadata.csv", index=False, encoding="utf-8-sig"
        )

        event_rows = []
        selected_threshold = (
            float(self.residual_model.rsplit("mad", 1)[1])
            if "event_mad" in self.residual_model
            else None
        )
        event_index = 0
        for event in self.residual_library.events:
            if selected_threshold is not None and event.threshold_mad != selected_threshold:
                continue
            for sample_index, value in enumerate(event.waveform_kw):
                event_rows.append(
                    {
                        "event_id": event_index,
                        "profile_id": event.profile_id,
                        "prompt_dataset": event.prompt_dataset,
                        "request_rate": event.request_rate,
                        "threshold_mad": event.threshold_mad,
                        "sign": event.sign,
                        "duration_s": event.duration_s,
                        "peak_kw": event.peak_kw,
                        "sample_index": sample_index,
                        "waveform_kw": value,
                    }
                )
            event_index += 1
        event_table = pd.DataFrame(event_rows)
        event_table.to_parquet(output / "event_waveform_library.parquet", index=False)

        manifest = {
            "artifact_type": "NLR prompt-conditioned M3 plus residual generator",
            "artifact_version": 1,
            "residual_model": self.residual_model,
            "prompt_datasets": list(self.prompt_datasets),
            "calibration_profile_count": len(self.residual_library.residuals),
            "event_template_count": int(event_index),
            "causal_plant_time_constant_claimed": False,
            "measurement_scope": "NLR job-total IT power; not a documented one-server unit",
            "aggregation_interpretation": (
                "n_effective is statistical diversity, while target_it_mean_mw fixes mean IT power"
            ),
            **self.artifact_metadata,
        }
        (output / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return output

    @classmethod
    def from_artifact(cls, artifact_dir: str | Path) -> "WorkloadGenerator":
        artifact = Path(artifact_dir).expanduser().resolve()
        manifest = json.loads((artifact / "artifact_manifest.json").read_text(encoding="utf-8"))
        parameter_table = pd.read_csv(artifact / "m3_parameters.csv")
        m3_parameters = {
            str(row.prompt_dataset): M3Parameters(
                p0_kw=float(row.p0_kw),
                delta_p_kw=float(row.delta_p_kw),
                k=float(row.k),
                x0=float(row.x0),
            )
            for row in parameter_table.itertuples(index=False)
        }
        statistics = pd.read_parquet(artifact / "calibration_statistics.parquet")
        residual_table = pd.read_parquet(artifact / "residual_waveform_library.parquet")
        residuals = {
            str(profile_id): group.sort_values("sample_index")["residual_kw"].to_numpy(float)
            for profile_id, group in residual_table.groupby("profile_id", observed=True)
        }
        profile_table = pd.read_csv(artifact / "profile_metadata.csv")
        profiles = {}
        for row in profile_table.itertuples(index=False):
            residual = residuals[str(row.profile_id)]
            interval = float(row.interval_s)
            profiles[str(row.profile_id)] = PowerProfile(
                profile_id=str(row.profile_id),
                condition_id=str(row.condition_id),
                prompt_dataset=str(row.prompt_dataset),
                request_rate=float(row.request_rate),
                split=str(row.split),
                time_s=np.arange(len(residual)) * interval,
                power_kw=np.zeros(len(residual)),
                interval_s=interval,
                measurement_origin=str(row.measurement_origin),
            )
        event_table = pd.read_parquet(artifact / "event_waveform_library.parquet")
        events = []
        for _, group in event_table.groupby("event_id", observed=True):
            first = group.iloc[0]
            events.append(
                EventTemplate(
                    profile_id=str(first["profile_id"]),
                    prompt_dataset=str(first["prompt_dataset"]),
                    request_rate=float(first["request_rate"]),
                    threshold_mad=float(first["threshold_mad"]),
                    sign=str(first["sign"]),
                    waveform_kw=group.sort_values("sample_index")["waveform_kw"].to_numpy(float),
                    duration_s=float(first["duration_s"]),
                    peak_kw=float(first["peak_kw"]),
                )
            )
        library = ResidualLibrary(
            statistics=statistics,
            residuals=residuals,
            events=events,
            profiles=profiles,
        )
        return cls(
            m3_parameters=m3_parameters,
            residual_library=library,
            residual_model=str(manifest["residual_model"]),
            artifact_metadata=manifest,
        )


def build_generator_from_nlr(
    *,
    dataset_root: str | Path,
    manifest_path: str | Path,
    condition_target_path: str | Path,
    split_path: str | Path,
    selection_path: str | Path,
) -> WorkloadGenerator:
    profiles = load_profiles(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split_path=split_path,
    )
    targets = load_condition_targets(condition_target_path, split_path)
    m3_parameters = fit_m3_by_dataset(targets, {"train", "validation"})
    library = build_residual_library(
        profiles, m3_parameters, {"train", "validation"}
    )
    selection = json.loads(
        Path(selection_path).expanduser().resolve().read_text(encoding="utf-8")
    )
    return WorkloadGenerator(
        m3_parameters=m3_parameters,
        residual_library=library,
        residual_model=str(selection["selected_residual_model"]),
        artifact_metadata={
            "calibration_splits": ["train", "validation"],
            "held_out_test_rates": "81-100 request/s",
            "source_selection": selection,
        },
    )


def facility_profile_metrics(profile: pd.DataFrame) -> dict[str, float]:
    power = pd.to_numeric(profile["it_power_mw"], errors="coerce").to_numpy(float)
    time_s = pd.to_numeric(profile["time_s"], errors="coerce").to_numpy(float)
    valid = np.isfinite(power) & np.isfinite(time_s)
    power, time_s = power[valid], time_s[valid]
    ramp = np.diff(power) / np.diff(time_s)
    mean = float(np.mean(power))
    return {
        "mean_it_power_mw": mean,
        "std_it_power_mw": float(np.std(power, ddof=0)),
        "coefficient_of_variation": float(np.std(power, ddof=0) / mean),
        "minimum_it_power_mw": float(np.min(power)),
        "maximum_it_power_mw": float(np.max(power)),
        "peak_to_average_ratio": float(np.max(power) / mean),
        "p01_it_power_mw": float(np.quantile(power, 0.01)),
        "p99_it_power_mw": float(np.quantile(power, 0.99)),
        "p95_abs_ramp_mw_per_s": float(np.quantile(np.abs(ramp), 0.95)),
        "maximum_ramp_up_mw_per_s": float(np.max(ramp)),
        "maximum_ramp_down_mw_per_s": float(np.min(ramp)),
    }


def run_aggregation_case_study(
    *,
    generator: WorkloadGenerator,
    output_dir: str | Path,
    target_it_mean_mw: float = DEFAULT_TARGET_IT_MW,
    request_rates: Sequence[float] = DEFAULT_REQUEST_RATES,
    effective_units: Sequence[int] = DEFAULT_EFFECTIVE_UNITS,
    common_mode_fractions: Sequence[float] = DEFAULT_COMMON_MODE_FRACTIONS,
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
    artifact = generator.save_artifact(output / "workload_generator_artifact")

    scenario_rows: list[dict[str, Any]] = []
    representative: list[pd.DataFrame] = []
    scenarios = [
        (dataset, float(rate), int(n_eff), float(rho), draw)
        for dataset in generator.prompt_datasets
        for rate in request_rates
        for n_eff in effective_units
        for rho in common_mode_fractions
        for draw in range(draws)
    ]
    for _, (dataset, rate, n_eff, rho, draw) in enumerate(
        tqdm(scenarios, desc="Facility aggregation scenarios")
    ):
        # Common random numbers across N_eff isolate the diversity effect.  In
        # particular rho=1 must produce the same profile for every N_eff.
        scenario_seed = int(
            (seed + zlib.crc32(f"{dataset}|{rate}|{rho}|{draw}".encode("utf-8")))
            % (2**32 - 1)
        )
        profile = generator.generate_facility_profile(
            prompt_dataset=dataset,
            request_rate=rate,
            target_it_mean_mw=target_it_mean_mw,
            n_effective=n_eff,
            common_mode_fraction=rho,
            duration_s=duration_s,
            interval_s=interval_s,
            seed=scenario_seed,
        )
        scenario_rows.append(
            {
                "prompt_dataset": dataset,
                "request_rate": rate,
                "target_it_mean_mw": target_it_mean_mw,
                "n_effective": n_eff,
                "common_mode_fraction": rho,
                "draw": draw,
                "theoretical_diversity_factor": float(
                    profile["theoretical_diversity_factor"].iloc[0]
                ),
                "illustrative_equivalent_trace_count": float(
                    target_it_mean_mw * 1000.0
                    / profile["equivalent_unit_steady_kw"].iloc[0]
                ),
                **facility_profile_metrics(profile),
            }
        )
        if rate == max(request_rates) and n_eff == 50 and draw == 0:
            representative.append(profile)

    scenario = pd.DataFrame(scenario_rows)
    group_columns = [
        "prompt_dataset",
        "request_rate",
        "target_it_mean_mw",
        "n_effective",
        "common_mode_fraction",
        "theoretical_diversity_factor",
    ]
    metric_columns = [
        "coefficient_of_variation",
        "std_it_power_mw",
        "peak_to_average_ratio",
        "p95_abs_ramp_mw_per_s",
        "maximum_ramp_up_mw_per_s",
        "maximum_ramp_down_mw_per_s",
        "minimum_it_power_mw",
        "maximum_it_power_mw",
    ]
    summary = (
        scenario.groupby(group_columns, observed=True)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        column if isinstance(column, str) else "_".join(str(x) for x in column if x)
        for column in summary.columns.to_flat_index()
    ]
    representative_table = (
        pd.concat(representative, ignore_index=True)
        if representative
        else pd.DataFrame()
    )
    scenario.to_csv(output / "aggregation_scenario_metrics.csv", index=False, encoding="utf-8-sig")
    scenario.to_parquet(output / "aggregation_scenario_metrics.parquet", index=False)
    summary.to_csv(output / "aggregation_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_parquet(output / "aggregation_summary.parquet", index=False)
    representative_table.to_csv(
        output / "representative_facility_profiles.csv", index=False, encoding="utf-8-sig"
    )
    if not representative_table.empty:
        representative_table.to_parquet(
            output / "representative_facility_profiles.parquet", index=False
        )

    _plot_diversity_curves(summary, max(request_rates), figures / "diversity_cv_curves.png")
    _plot_ramp_curves(summary, max(request_rates), figures / "diversity_ramp_curves.png")
    _plot_representative_profiles(
        representative_table, max(request_rates), figures / "representative_20mw_profiles.png"
    )

    report = _aggregation_report(
        summary=summary,
        target_it_mean_mw=target_it_mean_mw,
        max_rate=max(request_rates),
        draws=draws,
        artifact=artifact,
    )
    (output / "workload_aggregation_report_ko.md").write_text(report, encoding="utf-8")
    decision = {
        "generator_model": generator.residual_model,
        "target_power_interpretation": "IT mean power, not PCC total facility demand",
        "target_it_mean_mw": target_it_mean_mw,
        "n_effective_interpretation": "statistical diversity parameter, not physical server count",
        "common_mode_fraction_interpretation": "scenario parameter; not identified from current NLR traces",
        "scenario_count": len(scenario),
        "draws_per_scenario": draws,
        "artifact_path": str(artifact),
    }
    (output / "aggregation_methodology_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "scenario_metrics": scenario,
        "summary": summary,
        "decision": decision,
        "output_dir": output,
    }


def _plot_diversity_curves(summary: pd.DataFrame, rate: float, output: Path) -> None:
    selected = summary.loc[summary["request_rate"].eq(rate)].copy()
    datasets = list(selected["prompt_dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), dpi=150, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets, strict=True):
        part = selected.loc[selected["prompt_dataset"].eq(dataset)]
        for rho, group in part.groupby("common_mode_fraction", observed=True):
            ordered = group.sort_values("n_effective")
            ax.plot(
                ordered["n_effective"],
                ordered["coefficient_of_variation_mean"],
                marker="o",
                label=f"rho={rho:g}",
            )
        ax.set_xscale("log")
        ax.set_xlabel("Effective independent units")
        ax.set_title(str(dataset))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Facility IT power coefficient of variation")
    fig.suptitle(f"Diversity and common-mode sensitivity at {rate:g} request/s")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_ramp_curves(summary: pd.DataFrame, rate: float, output: Path) -> None:
    selected = summary.loc[summary["request_rate"].eq(rate)].copy()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for (dataset, rho), group in selected.groupby(
        ["prompt_dataset", "common_mode_fraction"], observed=True
    ):
        ordered = group.sort_values("n_effective")
        ax.plot(
            ordered["n_effective"],
            ordered["p95_abs_ramp_mw_per_s_mean"],
            marker="o",
            label=f"{dataset} | rho={rho:g}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Effective independent units")
    ax.set_ylabel("P95 absolute IT ramp (MW/s)")
    ax.set_title(f"Facility ramp sensitivity at {rate:g} request/s")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_representative_profiles(table: pd.DataFrame, rate: float, output: Path) -> None:
    if table.empty:
        return
    datasets = list(table["prompt_dataset"].drop_duplicates())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 4.2 * len(datasets)), dpi=150, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets, strict=True):
        part = table.loc[table["prompt_dataset"].eq(dataset)]
        for rho, group in part.groupby("common_mode_fraction", observed=True):
            ax.plot(group["time_s"], group["it_power_mw"], lw=1.0, label=f"rho={rho:g}")
        ax.set_title(f"{dataset} | {rate:g} request/s | N_eff=50")
        ax.set_ylabel("IT power (MW)")
        ax.grid(alpha=0.23)
        ax.legend(fontsize=8, ncol=4)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    values = frame.astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in values.to_numpy()]
    return "\n".join([header, rule, *rows])


def _aggregation_report(
    *,
    summary: pd.DataFrame,
    target_it_mean_mw: float,
    max_rate: float,
    draws: int,
    artifact: Path,
) -> str:
    selected = summary.loc[
        summary["request_rate"].eq(max_rate)
        & summary["n_effective"].isin({1, 50, 200})
        & summary["common_mode_fraction"].isin({0.0, 0.25, 1.0})
    ][
        [
            "prompt_dataset",
            "n_effective",
            "common_mode_fraction",
            "theoretical_diversity_factor",
            "coefficient_of_variation_mean",
            "p95_abs_ramp_mw_per_s_mean",
            "maximum_it_power_mw_mean",
            "minimum_it_power_mw_mean",
        ]
    ].round(5)
    return f"""# NLR workload generator 및 20 MW IT aggregation

## 구현 결과

- Generator: prompt-conditioned M3 + R3 AR(1)/event 3-MAD
- Target: **{target_it_mean_mw:g} MW 평균 IT power**
- 분석시간: 180 s, 해상도 0.1 s
- Monte Carlo: scenario당 {draws}회
- Portable artifact: `{artifact}`

## Aggregation 식

`e_agg = sqrt(rho) e_common + sqrt(1-rho) mean(e_independent)`

- `N_eff`: 비동기 workload 다양성을 나타내는 통계적 유효 unit 수
- `rho`: 여러 unit에 동시에 나타나는 common-mode 변동의 비율
- 이론적 residual 표준편차 비율: `sqrt(rho + (1-rho)/N_eff)`

## {max_rate:g} request/s 대표 결과

{_markdown_table(selected.reset_index(drop=True))}

## 해석

1. rho=0이면 N_eff 증가에 따라 독립적인 변동과 ramp가 평균화됨.
2. rho>0이면 N_eff를 늘려도 common-mode 변동이 남아 diversity 효과에 하한이 생김.
3. rho=1은 한 profile을 전체 설비에 동기적으로 곱하는 기존 단순 확대와 대응함.
4. 현재 NLR만으로 실제 facility rho 또는 N_eff를 식별할 수 없으므로 sensitivity scenario로 사용해야 함.

## 중요한 경계

- NLR profile은 job-total IT power이며 한 물리 서버의 대표 trace라고 확인되지 않음.
- `N_eff`는 실제 서버 대수가 아니며, 평균전력 크기는 target_it_mean_mw로 별도 설정함.
- 현재 20 MW는 IT 부하만 의미함. 향후 냉각, UPS loss 및 보조부하를 결합해야 PCC 부하가 됨.
- 수천 개 GPU가 동시에 같은 inference command를 수행하는 경우 rho가 커질 수 있으므로 완전 독립 aggregation만 사용하면 ramp를 과소평가할 수 있음.
"""
