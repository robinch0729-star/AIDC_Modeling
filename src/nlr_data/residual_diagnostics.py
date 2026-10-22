"""Diagnose whether NLR M3 residuals require white-noise, OU, or Jump-OU dynamics.

This stage deliberately precedes residual-model selection.  It fits the
prompt-conditioned M3 steady-state map on train conditions only, removes the
condition mean, fits a preliminary AR(1) (the sampled OU equivalent), and then
examines the remaining one-step innovations.

MAD thresholding is intentionally absent.  A three-component Gaussian mixture
is used only as a distributional diagnostic for a central background component
and signed tail components.  The result indicates whether Jump-OU should remain
in the later validation comparison; it is not itself the final Jump-OU
calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.special import logsumexp
from scipy.stats import chi2, expon, kstest, kurtosis, norm, skew
from tqdm.auto import tqdm

from .residual_models import (
    PowerProfile,
    fit_m3_by_dataset,
    load_condition_targets,
    load_profiles,
)


ACF_LAGS_S = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)


@dataclass(frozen=True)
class GaussianMixtureFit:
    weights: np.ndarray
    means: np.ndarray
    standard_deviations: np.ndarray
    responsibilities: np.ndarray
    log_likelihood: float
    bic: float
    converged: bool
    iterations: int
    background_component: int


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if lag <= 0:
        return 1.0
    if len(x) <= lag + 2:
        return float("nan")
    x = x - np.mean(x)
    denominator = float(np.dot(x, x))
    if denominator <= np.finfo(float).eps:
        return float("nan")
    return float(np.dot(x[:-lag], x[lag:]) / denominator)


def _ljung_box(values: np.ndarray, lags: int) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    lags = int(min(max(lags, 1), max(len(x) // 5, 1)))
    correlations = np.array([_autocorrelation(x, lag) for lag in range(1, lags + 1)])
    correlations = np.nan_to_num(correlations, nan=0.0)
    n = len(x)
    denominator = np.maximum(n - np.arange(1, lags + 1), 1)
    statistic = float(n * (n + 2) * np.sum(np.square(correlations) / denominator))
    return statistic, float(chi2.sf(statistic, df=lags))


def fit_ar1_ou(values: np.ndarray, interval_s: float) -> dict[str, float]:
    """Fit zero-mean AR(1) and report its continuous-time OU equivalent."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = x - np.mean(x)
    denominator = float(np.dot(x[:-1], x[:-1]))
    phi = (
        float(np.clip(np.dot(x[:-1], x[1:]) / denominator, -0.995, 0.9995))
        if len(x) >= 4 and denominator > np.finfo(float).eps
        else 0.0
    )
    innovations = x[1:] - phi * x[:-1]
    innovation_std = float(np.std(innovations, ddof=0))
    stationary_std = float(np.std(x, ddof=0))
    if 0.0 < phi < 1.0:
        theta = float(-math.log(phi) / interval_s)
        correlation_time = 1.0 / theta
        diffusion_sigma = float(
            innovation_std
            * math.sqrt(2.0 * theta / max(1.0 - math.exp(-2.0 * theta * interval_s), 1e-12))
        )
    else:
        theta = float("nan")
        correlation_time = float("nan")
        diffusion_sigma = float("nan")
    return {
        "phi": phi,
        "theta_per_s": theta,
        "correlation_time_s": correlation_time,
        "stationary_std_kw": stationary_std,
        "innovation_std_kw": innovation_std,
        "ou_diffusion_kw_per_sqrt_s": diffusion_sigma,
        "innovations": innovations,
    }


def _normal_log_density(x: np.ndarray, mean: float, standard_deviation: float) -> np.ndarray:
    sigma = max(float(standard_deviation), 1e-9)
    return -0.5 * np.square((x - mean) / sigma) - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)


def fit_gaussian_mixture_1d(
    values: np.ndarray,
    *,
    components: int = 3,
    max_iterations: int = 500,
    tolerance: float = 1e-6,
) -> GaussianMixtureFit:
    """Fit a small one-dimensional Gaussian mixture with deterministic EM."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        raise ValueError("At least 30 innovations are required for mixture diagnostics")
    if components != 3:
        raise ValueError("This diagnostic expects negative/background/positive components")
    overall_std = max(float(np.std(x, ddof=0)), 1e-9)
    means = np.quantile(x, [0.10, 0.50, 0.90]).astype(float)
    standard_deviations = np.full(components, max(overall_std * 0.60, 1e-9))
    weights = np.full(components, 1.0 / components)
    variance_floor = max((overall_std * 0.02) ** 2, 1e-18)
    previous = -np.inf
    converged = False
    responsibilities = np.full((len(x), components), 1.0 / components)

    for iteration in range(1, max_iterations + 1):
        log_probability = np.column_stack(
            [
                math.log(max(weights[index], 1e-12))
                + _normal_log_density(x, means[index], standard_deviations[index])
                for index in range(components)
            ]
        )
        normalizer = logsumexp(log_probability, axis=1)
        log_likelihood = float(np.sum(normalizer))
        responsibilities = np.exp(log_probability - normalizer[:, None])
        effective = np.maximum(responsibilities.sum(axis=0), 1e-8)
        weights = effective / len(x)
        means = (responsibilities * x[:, None]).sum(axis=0) / effective
        variances = (
            responsibilities * np.square(x[:, None] - means[None, :])
        ).sum(axis=0) / effective
        standard_deviations = np.sqrt(np.maximum(variances, variance_floor))
        order = np.argsort(means)
        weights = weights[order]
        means = means[order]
        standard_deviations = standard_deviations[order]
        responsibilities = responsibilities[:, order]
        if np.isfinite(previous) and abs(log_likelihood - previous) <= tolerance * (
            1.0 + abs(previous)
        ):
            converged = True
            break
        previous = log_likelihood

    parameter_count = (components - 1) + components + components
    bic = parameter_count * math.log(len(x)) - 2.0 * log_likelihood
    # The background component should be central and relatively prevalent.
    central_score = np.abs(means - np.median(x)) / np.maximum(standard_deviations, 1e-9)
    central_score -= 0.25 * np.log(np.maximum(weights, 1e-12))
    background = int(np.argmin(central_score))
    return GaussianMixtureFit(
        weights=weights,
        means=means,
        standard_deviations=standard_deviations,
        responsibilities=responsibilities,
        log_likelihood=log_likelihood,
        bic=float(bic),
        converged=converged,
        iterations=iteration,
        background_component=background,
    )


def _event_runs(
    event_probability: np.ndarray,
    innovations: np.ndarray,
    *,
    interval_s: float,
    probability_threshold: float = 0.80,
) -> dict[str, Any]:
    mask = np.asarray(event_probability) >= probability_threshold
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = changes[::2], changes[1::2]
    onsets = starts.astype(float) * interval_s
    interarrival = np.diff(onsets)
    positive_amplitude: list[float] = []
    negative_amplitude: list[float] = []
    durations: list[float] = []
    for start, stop in zip(starts, stops):
        segment = innovations[start:stop]
        if segment.size == 0:
            continue
        peak = float(segment[np.argmax(np.abs(segment))])
        durations.append(float((stop - start) * interval_s))
        if peak >= 0:
            positive_amplitude.append(peak)
        else:
            negative_amplitude.append(peak)
    total_duration = max(float(len(innovations) * interval_s), interval_s)
    if len(interarrival) >= 4 and float(np.mean(interarrival)) > 0:
        normalized = interarrival / float(np.mean(interarrival))
        exponential_ks_statistic, exponential_ks_pvalue = kstest(normalized, expon.cdf)
        interarrival_cv = float(np.std(interarrival, ddof=1) / np.mean(interarrival))
    else:
        exponential_ks_statistic = float("nan")
        exponential_ks_pvalue = float("nan")
        interarrival_cv = float("nan")
    return {
        "mixture_event_count": int(len(starts)),
        "mixture_event_rate_per_s": float(len(starts) / total_duration),
        "mixture_event_sample_fraction": float(np.mean(mask)),
        "positive_event_count": int(len(positive_amplitude)),
        "negative_event_count": int(len(negative_amplitude)),
        "positive_event_mean_peak_kw": float(np.mean(positive_amplitude)) if positive_amplitude else 0.0,
        "negative_event_mean_peak_kw": float(np.mean(negative_amplitude)) if negative_amplitude else 0.0,
        "event_mean_duration_s": float(np.mean(durations)) if durations else 0.0,
        "event_interarrival_mean_s": float(np.mean(interarrival)) if len(interarrival) else float("nan"),
        "event_interarrival_cv": interarrival_cv,
        "event_interarrival_exponential_ks_statistic": float(exponential_ks_statistic),
        "event_interarrival_exponential_ks_pvalue": float(exponential_ks_pvalue),
    }


def diagnose_profile(
    profile: PowerProfile,
    steady_power_kw: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    raw_residual = np.asarray(profile.power_kw - steady_power_kw, dtype=float)
    residual = raw_residual - np.mean(raw_residual)
    ar = fit_ar1_ou(residual, profile.interval_s)
    innovations = np.asarray(ar.pop("innovations"), dtype=float)
    innovation_mean = float(np.mean(innovations))
    innovation_std = max(float(np.std(innovations, ddof=0)), 1e-12)
    gaussian_ll = float(np.sum(_normal_log_density(innovations, innovation_mean, innovation_std)))
    gaussian_bic = 2 * math.log(len(innovations)) - 2.0 * gaussian_ll
    mixture = fit_gaussian_mixture_1d(innovations)
    delta_bic = float(gaussian_bic - mixture.bic)
    event_probability = 1.0 - mixture.responsibilities[:, mixture.background_component]
    event = _event_runs(
        event_probability,
        innovations,
        interval_s=profile.interval_s,
    )

    residual_lags = {
        f"residual_acf_{str(lag).replace('.', 'p')}s": _autocorrelation(
            residual, max(1, int(round(lag / profile.interval_s)))
        )
        for lag in ACF_LAGS_S
    }
    innovation_lags = {
        f"innovation_acf_{str(lag).replace('.', 'p')}s": _autocorrelation(
            innovations, max(1, int(round(lag / profile.interval_s)))
        )
        for lag in ACF_LAGS_S
    }
    residual_ljung_q, residual_ljung_p = _ljung_box(
        residual, max(5, int(round(2.0 / profile.interval_s)))
    )
    innovation_ljung_q, innovation_ljung_p = _ljung_box(
        innovations, max(5, int(round(2.0 / profile.interval_s)))
    )
    standardized = (innovations - innovation_mean) / innovation_std
    tail_probability = float(np.mean(np.abs(standardized) > 3.0))
    gaussian_tail_probability = float(2.0 * norm.sf(3.0))
    tail_ratio = tail_probability / gaussian_tail_probability
    ramp = np.diff(profile.power_kw) / profile.interval_s
    window_samples = max(5, min(int(round(30.0 / profile.interval_s)), len(residual) // 4))
    first_window_mean = float(np.mean(residual[:window_samples]))
    last_window_mean = float(np.mean(residual[-window_samples:]))
    first_last_shift = last_window_mean - first_window_mean
    residual_std = max(float(np.std(residual, ddof=0)), 1e-12)
    segment_means = np.array(
        [float(np.mean(segment)) for segment in np.array_split(residual, 6)], dtype=float
    )
    segment_mean_range = float(np.max(segment_means) - np.min(segment_means))
    frequency, spectrum = welch(
        residual,
        fs=1.0 / profile.interval_s,
        nperseg=min(1024, len(residual)),
        detrend="linear",
    )
    positive_frequency = frequency > 0
    total_spectrum = float(np.sum(spectrum[positive_frequency]))
    low_frequency_share = (
        float(np.sum(spectrum[positive_frequency & (frequency <= 0.1)]) / total_spectrum)
        if total_spectrum > 0
        else 0.0
    )

    mixture_columns: dict[str, float] = {}
    for index in range(3):
        mixture_columns[f"gmm_weight_{index}"] = float(mixture.weights[index])
        mixture_columns[f"gmm_mean_{index}_kw"] = float(mixture.means[index])
        mixture_columns[f"gmm_std_{index}_kw"] = float(mixture.standard_deviations[index])

    row = {
        "profile_id": profile.profile_id,
        "condition_id": profile.condition_id,
        "prompt_dataset": profile.prompt_dataset,
        "request_rate": profile.request_rate,
        "split": profile.split,
        "interval_s": profile.interval_s,
        "duration_s": float(profile.time_s[-1]),
        "m3_steady_power_kw": steady_power_kw,
        "profile_mean_power_kw": float(np.mean(profile.power_kw)),
        "m3_condition_bias_kw": float(np.mean(raw_residual)),
        "residual_std_kw": residual_std,
        "first_last_30s_shift_kw": first_last_shift,
        "first_last_30s_shift_in_sigma": first_last_shift / residual_std,
        "six_segment_mean_range_kw": segment_mean_range,
        "six_segment_mean_range_in_sigma": segment_mean_range / residual_std,
        "nonstationary_mean_shift_evidence": bool(segment_mean_range / residual_std > 1.5),
        **ar,
        **residual_lags,
        **innovation_lags,
        "residual_ljung_box_q": residual_ljung_q,
        "residual_ljung_box_pvalue": residual_ljung_p,
        "innovation_ljung_box_q": innovation_ljung_q,
        "innovation_ljung_box_pvalue": innovation_ljung_p,
        "innovation_mean_kw": innovation_mean,
        "innovation_skewness": float(skew(innovations, bias=False)),
        "innovation_excess_kurtosis": float(kurtosis(innovations, fisher=True, bias=False)),
        "innovation_p001_kw": float(np.quantile(innovations, 0.001)),
        "innovation_p01_kw": float(np.quantile(innovations, 0.01)),
        "innovation_p99_kw": float(np.quantile(innovations, 0.99)),
        "innovation_p999_kw": float(np.quantile(innovations, 0.999)),
        "innovation_abs_gt_3sigma_fraction": tail_probability,
        "innovation_gaussian_tail_ratio": tail_ratio,
        "gaussian_innovation_bic": float(gaussian_bic),
        "gmm3_innovation_bic": float(mixture.bic),
        "gaussian_minus_gmm3_bic": delta_bic,
        "gmm_converged": bool(mixture.converged),
        "gmm_iterations": int(mixture.iterations),
        "gmm_background_component": int(mixture.background_component),
        **mixture_columns,
        **event,
        "p95_abs_ramp_kw_per_s": float(np.quantile(np.abs(ramp), 0.95)),
        "max_ramp_up_kw_per_s": float(np.max(ramp)),
        "max_ramp_down_kw_per_s": float(np.min(ramp)),
        "residual_low_frequency_psd_share": low_frequency_share,
        "r2_ou_evidence": bool(ar["phi"] > 0.10 and residual_ljung_p < 0.05),
        "r3_jump_evidence": bool(
            mixture.converged and delta_bic > 10.0 and tail_ratio > 1.5
        ),
    }
    trace = pd.DataFrame(
        {
            "profile_id": profile.profile_id,
            "prompt_dataset": profile.prompt_dataset,
            "request_rate": profile.request_rate,
            "split": profile.split,
            "time_s": profile.time_s,
            "measured_power_kw": profile.power_kw,
            "m3_steady_power_kw": steady_power_kw,
            "centered_residual_kw": residual,
            "ou_innovation_kw": np.concatenate(([np.nan], innovations)),
            "mixture_event_probability": np.concatenate(([np.nan], event_probability)),
        }
    )
    return row, trace


def aggregate_diagnostics(profile_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, dataset), group in profile_table.groupby(
        ["split", "prompt_dataset"], observed=True
    ):
        rows.append(
            {
                "split": split,
                "prompt_dataset": dataset,
                "profile_count": int(len(group)),
                "median_phi": float(group["phi"].median()),
                "median_correlation_time_s": float(group["correlation_time_s"].median()),
                "median_residual_std_kw": float(group["residual_std_kw"].median()),
                "median_innovation_std_kw": float(group["innovation_std_kw"].median()),
                "median_innovation_excess_kurtosis": float(group["innovation_excess_kurtosis"].median()),
                "median_gaussian_tail_ratio": float(group["innovation_gaussian_tail_ratio"].median()),
                "median_gaussian_minus_gmm3_bic": float(group["gaussian_minus_gmm3_bic"].median()),
                "median_mixture_event_rate_per_s": float(group["mixture_event_rate_per_s"].median()),
                "median_event_sample_fraction": float(group["mixture_event_sample_fraction"].median()),
                "median_six_segment_mean_range_in_sigma": float(
                    group["six_segment_mean_range_in_sigma"].median()
                ),
                "nonstationary_mean_shift_fraction": float(
                    np.mean(group["nonstationary_mean_shift_evidence"])
                ),
                "residual_serial_correlation_fraction": float(
                    np.mean(group["residual_ljung_box_pvalue"] < 0.05)
                ),
                "innovation_not_white_fraction": float(
                    np.mean(group["innovation_ljung_box_pvalue"] < 0.05)
                ),
                "r2_ou_evidence_fraction": float(np.mean(group["r2_ou_evidence"])) ,
                "r3_jump_evidence_fraction": float(np.mean(group["r3_jump_evidence"])) ,
                "gmm_convergence_fraction": float(np.mean(group["gmm_converged"])) ,
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "prompt_dataset"], kind="stable")


def structural_decision(profile_table: pd.DataFrame) -> dict[str, Any]:
    train = profile_table.loc[profile_table["split"].eq("train")].copy()
    if train.empty:
        raise ValueError("No train profiles available for structural decision")
    serial_fraction = float(np.mean(train["residual_ljung_box_pvalue"] < 0.05))
    median_phi = float(train["phi"].median())
    mixture_fraction = float(
        np.mean(train["gmm_converged"] & (train["gaussian_minus_gmm3_bic"] > 10.0))
    )
    jump_fraction = float(np.mean(train["r3_jump_evidence"]))
    median_tail_ratio = float(train["innovation_gaussian_tail_ratio"].median())
    convergence_fraction = float(np.mean(train["gmm_converged"]))
    innovation_not_white_fraction = float(
        np.mean(train["innovation_ljung_box_pvalue"] < 0.05)
    )
    nonstationary_fraction = float(np.mean(train["nonstationary_mean_shift_evidence"]))
    retain_ou = bool(serial_fraction >= 0.50 and median_phi > 0.10)
    retain_jump_ou = bool(mixture_fraction >= 0.50 and jump_fraction >= 0.50)
    candidates = ["R1_white_noise"]
    if retain_ou:
        candidates.append("R2_ou")
    if retain_jump_ou:
        candidates.append("R3_jump_ou")
    return {
        "decision_scope": "train split only",
        "retain_r1_white_noise_baseline": True,
        "retain_r2_ou_for_validation": retain_ou,
        "retain_r3_jump_ou_for_validation": retain_jump_ou,
        "validation_candidate_models": candidates,
        "train_profile_count": int(len(train)),
        "train_median_phi": median_phi,
        "train_residual_serial_correlation_fraction": serial_fraction,
        "train_gmm3_bic_improvement_fraction": mixture_fraction,
        "train_jump_evidence_fraction": jump_fraction,
        "train_median_gaussian_tail_ratio": median_tail_ratio,
        "train_gmm_convergence_fraction": convergence_fraction,
        "train_innovation_not_white_fraction": innovation_not_white_fraction,
        "train_nonstationary_mean_shift_fraction": nonstationary_fraction,
        "mad_role": "not used; optional initialization/sensitivity analysis only",
        "ou_caveat": (
            "High phi establishes temporal persistence, but mean-shift and innovation "
            "whiteness diagnostics must be checked before interpreting it as a complete OU model."
        ),
        "interpretation": (
            "This is a structural screening decision. Final residual-model selection "
            "must be performed on validation conditions."
        ),
    }


def _plot_representative_traces(traces: pd.DataFrame, output: Path) -> None:
    selected: list[pd.DataFrame] = []
    for dataset in traces["prompt_dataset"].drop_duplicates():
        part = traces.loc[
            traces["prompt_dataset"].eq(dataset)
            & traces["request_rate"].eq(50.0)
        ]
        if part.empty:
            candidates = traces.loc[traces["prompt_dataset"].eq(dataset)]
            nearest = (candidates["request_rate"] - 50.0).abs().idxmin()
            profile_id = candidates.loc[nearest, "profile_id"]
            part = candidates.loc[candidates["profile_id"].eq(profile_id)]
        selected.append(part)
    if not selected:
        return
    working = pd.concat(selected, ignore_index=True)
    datasets = working["prompt_dataset"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(datasets), 2, figsize=(14, 4.0 * len(datasets)), dpi=160)
    axes = np.asarray(axes).reshape(len(datasets), 2)
    for row_index, dataset in enumerate(datasets):
        part = working.loc[working["prompt_dataset"].eq(dataset)]
        left, right = axes[row_index]
        left.plot(part["time_s"], part["centered_residual_kw"], color="#111827", lw=0.8)
        left.axhline(0.0, color="#9ca3af", lw=0.8)
        left.set_title(f"{dataset}: centered M3 residual")
        left.set_ylabel("Residual (kW)")
        left.grid(alpha=0.2)
        event = part["mixture_event_probability"] >= 0.80
        right.plot(part["time_s"], part["ou_innovation_kw"], color="#2563eb", lw=0.75)
        right.scatter(
            part.loc[event, "time_s"],
            part.loc[event, "ou_innovation_kw"],
            s=10,
            color="#dc2626",
            label="GMM tail probability >= 0.8",
            zorder=3,
        )
        right.axhline(0.0, color="#9ca3af", lw=0.8)
        right.set_title(f"{dataset}: OU innovation (diagnostic)")
        right.set_ylabel("Innovation (kW)")
        right.grid(alpha=0.2)
        right.legend(loc="best", fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_diagnostic_map(profile_table: pd.DataFrame, output: Path) -> None:
    train = profile_table.loc[profile_table["split"].eq("train")].copy()
    datasets = train["prompt_dataset"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(datasets), 3, figsize=(15, 3.8 * len(datasets)), dpi=160)
    axes = np.asarray(axes).reshape(len(datasets), 3)
    for row_index, dataset in enumerate(datasets):
        part = train.loc[train["prompt_dataset"].eq(dataset)].sort_values("request_rate")
        metrics = (
            ("phi", "AR(1) phi", "#2563eb"),
            ("innovation_gaussian_tail_ratio", "Innovation >3sigma / Gaussian", "#ea580c"),
            ("gaussian_minus_gmm3_bic", "BIC(Gaussian)-BIC(GMM3)", "#059669"),
        )
        for axis, (column, title, color) in zip(axes[row_index], metrics, strict=True):
            axis.plot(part["request_rate"], part[column], marker="o", ms=3, lw=1.0, color=color)
            axis.set_title(f"{dataset}: {title}")
            axis.set_xlabel("Request rate (request/s)")
            axis.grid(alpha=0.25)
        axes[row_index, 0].axhline(0.10, color="#6b7280", ls="--", lw=0.8)
        axes[row_index, 1].axhline(1.0, color="#6b7280", ls="--", lw=0.8)
        axes[row_index, 2].axhline(10.0, color="#6b7280", ls="--", lw=0.8)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_innovation_qq(traces: pd.DataFrame, output: Path) -> None:
    selected = traces.loc[traces["request_rate"].eq(50.0)].copy()
    if selected.empty:
        selected_parts = []
        for dataset in traces["prompt_dataset"].drop_duplicates():
            candidates = traces.loc[traces["prompt_dataset"].eq(dataset)]
            nearest = (candidates["request_rate"] - 50.0).abs().idxmin()
            profile_id = candidates.loc[nearest, "profile_id"]
            selected_parts.append(candidates.loc[candidates["profile_id"].eq(profile_id)])
        selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    datasets = selected["prompt_dataset"].drop_duplicates().tolist()
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.0 * len(datasets), 5.0), dpi=160)
    axes = np.atleast_1d(axes)
    for axis, dataset in zip(axes, datasets, strict=True):
        values = selected.loc[
            selected["prompt_dataset"].eq(dataset), "ou_innovation_kw"
        ].dropna().to_numpy(float)
        values = np.sort((values - np.mean(values)) / max(np.std(values), 1e-12))
        probability = (np.arange(len(values)) + 0.5) / len(values)
        theoretical = norm.ppf(probability)
        axis.scatter(theoretical, values, s=6, alpha=0.55, color="#2563eb")
        limits = [min(theoretical.min(), values.min()), max(theoretical.max(), values.max())]
        axis.plot(limits, limits, color="#dc2626", ls="--", lw=1.0)
        axis.set_title(f"{dataset}: OU innovation Q-Q")
        axis.set_xlabel("Standard normal quantile")
        axis.set_ylabel("Observed standardized quantile")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        formatted = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{float(value):.4g}" if np.isfinite(value) else "NaN")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def _write_report(
    profile_table: pd.DataFrame,
    aggregate: pd.DataFrame,
    decision: Mapping[str, Any],
    output: Path,
) -> None:
    columns = [
        "split",
        "prompt_dataset",
        "profile_count",
        "median_phi",
        "median_correlation_time_s",
        "median_innovation_excess_kurtosis",
        "median_gaussian_tail_ratio",
        "median_gaussian_minus_gmm3_bic",
        "median_six_segment_mean_range_in_sigma",
        "nonstationary_mean_shift_fraction",
        "r2_ou_evidence_fraction",
        "r3_jump_evidence_fraction",
    ]
    text = f"""# NLR M3 residual 구조 진단

## 목적

- M3 정상상태식이 설명하지 못한 시간변동을 분리했다.
- train condition만 이용해 OU/AR(1)을 우선 적합하고 innovation을 계산했다.
- R1 white noise, R2 OU, R3 Jump-OU 중 validation에서 비교할 구조를 선별했다.
- MAD 임계값은 사용하지 않았다. 3-component Gaussian mixture는 Jump-OU 필요성을 확인하기 위한 분포 진단일 뿐 최종 모델은 아니다.

## 구조 선별 결과

```json
{json.dumps(dict(decision), indent=2, ensure_ascii=False)}
```

## 집계 결과

{_markdown_table(aggregate[columns])}

## 해석 원칙

- residual에 유의한 serial correlation이 반복되면 R1만으로 충분하지 않으므로 R2 OU를 유지한다.
- OU innovation이 Gaussian보다 뚜렷한 tail mixture를 가지면 R3 Jump-OU를 validation 후보로 유지한다.
- 이 단계는 후보를 탈락시키는 구조 진단이다. 최종 선택은 validation condition에서 ACF, PSD, ramp 및 tail 재현성을 비교해 수행한다.
- `mixture_event_probability`는 진단용 posterior이며, 실제 jump 발생시각의 확정 label로 해석하지 않는다.

## 산출물

- `profile_residual_diagnostics.csv`: condition별 전체 진단지표
- `aggregate_residual_diagnostics.csv`: split·prompt dataset별 집계
- `representative_residual_innovation.csv`: 대표 figure용 trace
- `residual_innovation_representative.png`: residual과 OU innovation
- `train_structure_diagnostic_map.png`: request rate별 구조 지표
- `innovation_normal_qq.png`: Gaussian innovation 가정 점검
"""
    output.write_text(text, encoding="utf-8")


def run_residual_diagnostics(
    *,
    dataset_root: str | Path,
    manifest_path: str | Path,
    condition_target_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split_path=split_path,
    )
    targets = load_condition_targets(condition_target_path, split_path)
    m3_parameters = fit_m3_by_dataset(targets, {"train"})
    rows: list[dict[str, Any]] = []
    trace_parts: list[pd.DataFrame] = []
    for profile in tqdm(profiles, desc="Diagnosing M3 residuals"):
        steady = float(m3_parameters[profile.prompt_dataset].predict(profile.request_rate))
        row, trace = diagnose_profile(profile, steady)
        rows.append(row)
        if profile.request_rate in {10.0, 50.0, 90.0, 100.0}:
            trace_parts.append(trace)
    profile_table = pd.DataFrame(rows).sort_values(
        ["split", "prompt_dataset", "request_rate"], kind="stable"
    )
    traces = pd.concat(trace_parts, ignore_index=True) if trace_parts else pd.DataFrame()
    aggregate = aggregate_diagnostics(profile_table)
    decision = structural_decision(profile_table)

    profile_table.to_csv(output / "profile_residual_diagnostics.csv", index=False)
    aggregate.to_csv(output / "aggregate_residual_diagnostics.csv", index=False)
    traces.to_csv(output / "representative_residual_innovation.csv", index=False)
    (output / "residual_structure_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not traces.empty:
        _plot_representative_traces(traces, output / "residual_innovation_representative.png")
        _plot_innovation_qq(traces, output / "innovation_normal_qq.png")
    _plot_diagnostic_map(profile_table, output / "train_structure_diagnostic_map.png")
    _write_report(profile_table, aggregate, decision, output / "residual_diagnostics_report_ko.md")
    return {
        "output_dir": output,
        "profile_diagnostics": profile_table,
        "aggregate_diagnostics": aggregate,
        "decision": decision,
    }
