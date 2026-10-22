"""Select an NLR residual generator without MAD event thresholding.

Candidate equations:

* R0: M3 only.
* R1: Gaussian white-noise residual.
* R2: sampled OU / AR(1) with Gaussian innovation.
* R3: sampled Jump-OU approximated by a signed three-component innovation
  mixture (background, negative jump, positive jump).
* R4: empirical train-trace replay, reference only.

The M3 map and residual parameters are calibrated without using the evaluation
split.  Validation selects R1--R3.  The selected family is then recalibrated on
train+validation and evaluated on the high-rate test split.
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
from scipy.signal import lfilter, welch
from scipy.stats import kurtosis, norm, wasserstein_distance
from tqdm.auto import tqdm

from .dynamic_identifiability import _autocorrelation
from .residual_diagnostics import fit_ar1_ou, fit_gaussian_mixture_1d
from .residual_models import (
    PowerProfile,
    fit_m3_by_dataset,
    load_condition_targets,
    load_profiles,
)


MODEL_NAMES = (
    "R0_m3_only",
    "R1_white_noise",
    "R2_ou",
    "R3_jump_ou",
    "R4_empirical_replay",
)
DEPLOYABLE_MODELS = ("R1_white_noise", "R2_ou", "R3_jump_ou")
ACF_LAGS_S = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
ERROR_COMPONENTS = (
    "variability_error",
    "distribution_error",
    "acf_error",
    "ramp_error",
    "spectral_error",
    "innovation_tail_error",
    "innovation_acf_error",
)


@dataclass
class ResidualParameterLibrary:
    parameters: pd.DataFrame
    residuals: dict[str, np.ndarray]

    def neighbours(
        self, prompt_dataset: str, request_rate: float, *, k: int = 8
    ) -> tuple[pd.DataFrame, np.ndarray]:
        part = self.parameters.loc[
            self.parameters["prompt_dataset"].eq(prompt_dataset)
        ].copy()
        if part.empty:
            raise KeyError(f"No residual parameters for {prompt_dataset}")
        part["distance"] = np.abs(part["request_rate"] - request_rate)
        part = part.sort_values(["distance", "request_rate"], kind="stable").head(k)
        distance = part["distance"].to_numpy(float)
        weight = 1.0 / np.maximum(distance, 0.5)
        return part, weight / weight.sum()

    def weighted(self, prompt_dataset: str, request_rate: float, column: str) -> float:
        part, weight = self.neighbours(prompt_dataset, request_rate)
        values = pd.to_numeric(part[column], errors="coerce").to_numpy(float)
        valid = np.isfinite(values)
        if not valid.any():
            return float("nan")
        return float(np.average(values[valid], weights=weight[valid] / weight[valid].sum()))


def _canonical_mixture(fit: Any) -> dict[str, float]:
    background = int(fit.background_component)
    others = [index for index in range(3) if index != background]
    negative = min(others, key=lambda index: fit.means[index])
    positive = max(others, key=lambda index: fit.means[index])
    if negative == positive:
        negative, positive = others
    result: dict[str, float] = {}
    for role, index in (("negative", negative), ("background", background), ("positive", positive)):
        result[f"{role}_weight"] = float(fit.weights[index])
        result[f"{role}_mean_kw"] = float(fit.means[index])
        result[f"{role}_std_kw"] = float(fit.standard_deviations[index])
    return result


def build_parameter_library(
    profiles: Sequence[PowerProfile],
    m3_parameters: Mapping[str, Any],
    calibration_splits: Iterable[str],
) -> ResidualParameterLibrary:
    allowed = set(calibration_splits)
    rows: list[dict[str, Any]] = []
    residuals: dict[str, np.ndarray] = {}
    for profile in profiles:
        if profile.split not in allowed:
            continue
        steady = float(m3_parameters[profile.prompt_dataset].predict(profile.request_rate))
        residual = np.asarray(profile.power_kw - steady, dtype=float)
        residual = residual - np.mean(residual)
        ar = fit_ar1_ou(residual, profile.interval_s)
        innovations = np.asarray(ar.pop("innovations"), dtype=float)
        mixture = fit_gaussian_mixture_1d(innovations)
        innovation_mean = float(np.mean(innovations))
        innovation_std = max(float(np.std(innovations, ddof=0)), 1e-12)
        gaussian_log_likelihood = float(
            np.sum(norm.logpdf(innovations, loc=innovation_mean, scale=innovation_std))
        )
        gaussian_bic = 2.0 * math.log(len(innovations)) - 2.0 * gaussian_log_likelihood
        standardized = (innovations - innovation_mean) / innovation_std
        tail_ratio = float(
            np.mean(np.abs(standardized) > 3.0) / (2.0 * norm.sf(3.0))
        )
        bic_improvement = float(gaussian_bic - mixture.bic)
        jump_evidence = bool(
            mixture.converged and bic_improvement > 10.0 and tail_ratio > 1.5
        )
        residuals[profile.profile_id] = residual
        rows.append(
            {
                "profile_id": profile.profile_id,
                "condition_id": profile.condition_id,
                "prompt_dataset": profile.prompt_dataset,
                "request_rate": profile.request_rate,
                "split": profile.split,
                "interval_s": profile.interval_s,
                "residual_std_kw": float(np.std(residual, ddof=0)),
                "jump_evidence": float(jump_evidence),
                "gaussian_minus_gmm3_bic": bic_improvement,
                "innovation_gaussian_tail_ratio": tail_ratio,
                **{key: value for key, value in ar.items() if isinstance(value, (int, float))},
                **_canonical_mixture(mixture),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(f"No calibration profiles for {sorted(allowed)}")
    return ResidualParameterLibrary(table, residuals)


def _simulate_ar(
    length: int,
    *,
    phi: float,
    innovation: np.ndarray,
    initial_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.empty(length, dtype=float)
    result[0] = rng.normal(0.0, max(initial_std, 0.0))
    for index in range(1, length):
        result[index] = phi * result[index - 1] + innovation[index - 1]
    return result - np.mean(result)


def simulate_residual(
    model_name: str,
    *,
    length: int,
    prompt_dataset: str,
    request_rate: float,
    library: ResidualParameterLibrary,
    rng: np.random.Generator,
) -> np.ndarray:
    if model_name == "R0_m3_only":
        return np.zeros(length)
    part, weights = library.neighbours(prompt_dataset, request_rate)
    residual_std = library.weighted(prompt_dataset, request_rate, "residual_std_kw")
    if model_name == "R1_white_noise":
        result = rng.normal(0.0, max(residual_std, 0.0), length)
    elif model_name == "R2_ou":
        phi = float(np.clip(library.weighted(prompt_dataset, request_rate, "phi"), 0.0, 0.9995))
        innovation_std = max(
            library.weighted(prompt_dataset, request_rate, "innovation_std_kw"), 1e-12
        )
        innovations = rng.normal(0.0, innovation_std, length - 1)
        initial_std = innovation_std / math.sqrt(max(1.0 - phi**2, 1e-8))
        result = _simulate_ar(
            length,
            phi=phi,
            innovation=innovations,
            initial_std=initial_std,
            rng=rng,
        )
    elif model_name == "R3_jump_ou":
        phi = float(np.clip(library.weighted(prompt_dataset, request_rate, "phi"), 0.0, 0.9995))
        jump_gate = float(
            np.clip(library.weighted(prompt_dataset, request_rate, "jump_evidence"), 0.0, 1.0)
        )
        gaussian_innovation_std = max(
            library.weighted(prompt_dataset, request_rate, "innovation_std_kw"), 1e-12
        )
        roles = ("negative", "background", "positive")
        mixture_weights = np.array(
            [library.weighted(prompt_dataset, request_rate, f"{role}_weight") for role in roles]
        )
        mixture_weights = np.maximum(mixture_weights, 1e-8)
        mixture_weights /= mixture_weights.sum()
        means = np.array(
            [library.weighted(prompt_dataset, request_rate, f"{role}_mean_kw") for role in roles]
        )
        standard_deviations = np.array(
            [library.weighted(prompt_dataset, request_rate, f"{role}_std_kw") for role in roles]
        )
        # Preserve zero-mean residual innovations after interpolating conditions.
        means = means - float(np.dot(mixture_weights, means))
        component = rng.choice(3, size=length - 1, p=mixture_weights)
        mixture_innovations = rng.normal(means[component], standard_deviations[component])
        use_jump_mixture = rng.random(length - 1) < jump_gate
        gaussian_innovations = rng.normal(0.0, gaussian_innovation_std, length - 1)
        innovations = np.where(use_jump_mixture, mixture_innovations, gaussian_innovations)
        mixture_variance = float(
            np.dot(mixture_weights, np.square(standard_deviations) + np.square(means))
        )
        innovation_variance = (
            jump_gate * mixture_variance
            + (1.0 - jump_gate) * gaussian_innovation_std**2
        )
        initial_std = math.sqrt(innovation_variance / max(1.0 - phi**2, 1e-8))
        result = _simulate_ar(
            length,
            phi=phi,
            innovation=innovations,
            initial_std=initial_std,
            rng=rng,
        )
    elif model_name == "R4_empirical_replay":
        donor_index = int(rng.choice(len(part), p=weights))
        donor_id = str(part.iloc[donor_index]["profile_id"])
        donor = np.asarray(library.residuals[donor_id], dtype=float)
        if len(donor) < length:
            donor = np.tile(donor, int(np.ceil(length / len(donor))))
        result = np.roll(donor, int(rng.integers(0, len(donor))))[:length]
    else:
        raise KeyError(model_name)
    return np.asarray(result, dtype=float) - float(np.mean(result))


def simulate_residual_ensemble(
    model_name: str,
    *,
    draws: int,
    length: int,
    prompt_dataset: str,
    request_rate: float,
    library: ResidualParameterLibrary,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate an independent residual ensemble without a Python time loop.

    This is distribution-equivalent to repeated ``simulate_residual`` calls for
    R0--R3, although the exact random-number ordering is intentionally not
    preserved.  Each row is demeaned in the same way as the scalar generator.
    """
    if draws < 1 or length < 1:
        raise ValueError("draws and length must be positive")
    if model_name == "R0_m3_only":
        return np.zeros((draws, length), dtype=float)

    residual_std = library.weighted(prompt_dataset, request_rate, "residual_std_kw")
    if model_name == "R1_white_noise":
        result = rng.normal(0.0, max(residual_std, 0.0), size=(draws, length))
    elif model_name in {"R2_ou", "R3_jump_ou"}:
        phi = float(
            np.clip(library.weighted(prompt_dataset, request_rate, "phi"), 0.0, 0.9995)
        )
        gaussian_innovation_std = max(
            library.weighted(prompt_dataset, request_rate, "innovation_std_kw"), 1e-12
        )
        if model_name == "R2_ou":
            innovations = rng.normal(
                0.0, gaussian_innovation_std, size=(draws, max(length - 1, 0))
            )
            innovation_variance = gaussian_innovation_std**2
        else:
            jump_gate = float(
                np.clip(
                    library.weighted(prompt_dataset, request_rate, "jump_evidence"),
                    0.0,
                    1.0,
                )
            )
            roles = ("negative", "background", "positive")
            mixture_weights = np.array(
                [
                    library.weighted(prompt_dataset, request_rate, f"{role}_weight")
                    for role in roles
                ],
                dtype=float,
            )
            mixture_weights = np.maximum(mixture_weights, 1e-8)
            mixture_weights /= mixture_weights.sum()
            means = np.array(
                [
                    library.weighted(prompt_dataset, request_rate, f"{role}_mean_kw")
                    for role in roles
                ],
                dtype=float,
            )
            standard_deviations = np.array(
                [
                    library.weighted(prompt_dataset, request_rate, f"{role}_std_kw")
                    for role in roles
                ],
                dtype=float,
            )
            # Keep the interpolated mixture innovation centered at zero.
            means = means - float(np.dot(mixture_weights, means))
            shape = (draws, max(length - 1, 0))
            component = rng.choice(3, size=shape, p=mixture_weights)
            mixture_innovations = rng.normal(
                means[component], standard_deviations[component]
            )
            use_jump_mixture = rng.random(shape) < jump_gate
            gaussian_innovations = rng.normal(
                0.0, gaussian_innovation_std, size=shape
            )
            innovations = np.where(
                use_jump_mixture, mixture_innovations, gaussian_innovations
            )
            mixture_variance = float(
                np.dot(
                    mixture_weights,
                    np.square(standard_deviations) + np.square(means),
                )
            )
            innovation_variance = (
                jump_gate * mixture_variance
                + (1.0 - jump_gate) * gaussian_innovation_std**2
            )

        initial_std = math.sqrt(innovation_variance / max(1.0 - phi**2, 1e-8))
        driving = np.empty((draws, length), dtype=float)
        driving[:, 0] = rng.normal(0.0, initial_std, size=draws)
        if length > 1:
            driving[:, 1:] = innovations
        # y[k] = driving[k] + phi*y[k-1]: sampled OU/AR(1) recurrence.
        result = lfilter([1.0], [1.0, -phi], driving, axis=1)
    else:
        raise KeyError(
            f"Vectorized ensemble generation supports R0-R3, not {model_name}"
        )

    result = np.asarray(result, dtype=float)
    return result - np.mean(result, axis=1, keepdims=True)


def temporal_features(values: np.ndarray, interval_s: float) -> dict[str, float]:
    residual = np.asarray(values, dtype=float)
    residual = residual[np.isfinite(residual)]
    residual = residual - np.mean(residual)
    if len(residual) < 30:
        raise ValueError("At least 30 residual samples are required")
    std = max(float(np.std(residual, ddof=0)), 1e-12)
    ramp = np.diff(residual) / interval_s
    ar = fit_ar1_ou(residual, interval_s)
    innovations = np.asarray(ar.pop("innovations"), dtype=float)
    innovation_std = max(float(np.std(innovations, ddof=0)), 1e-12)
    standardized = (innovations - np.mean(innovations)) / innovation_std
    frequency, spectrum = welch(
        residual,
        fs=1.0 / interval_s,
        nperseg=min(1024, len(residual)),
        detrend="linear",
    )
    positive = frequency > 0
    total = float(np.sum(spectrum[positive]))
    innovation_kurtosis = float(kurtosis(innovations, fisher=True, bias=False))
    if not np.isfinite(innovation_kurtosis):
        innovation_kurtosis = 0.0
    result = {
        "std_kw": std,
        "p01_kw": float(np.quantile(residual, 0.01)),
        "p05_kw": float(np.quantile(residual, 0.05)),
        "p95_kw": float(np.quantile(residual, 0.95)),
        "p99_kw": float(np.quantile(residual, 0.99)),
        "p95_abs_ramp_kw_per_s": float(np.quantile(np.abs(ramp), 0.95)),
        "p99_abs_ramp_kw_per_s": float(np.quantile(np.abs(ramp), 0.99)),
        "low_frequency_psd_share": (
            float(np.sum(spectrum[positive & (frequency <= 0.1)]) / total) if total > 0 else 0.0
        ),
        "innovation_excess_kurtosis": innovation_kurtosis,
        "innovation_gaussian_tail_ratio": float(
            np.mean(np.abs(standardized) > 3.0) / (2.0 * norm.sf(3.0))
        ),
    }
    for lag in ACF_LAGS_S:
        residual_acf = _autocorrelation(
            residual, max(1, int(round(lag / interval_s)))
        )
        innovation_acf = _autocorrelation(
            innovations, max(1, int(round(lag / interval_s)))
        )
        result[f"acf_{str(lag).replace('.', 'p')}s"] = (
            float(residual_acf) if np.isfinite(residual_acf) else 0.0
        )
        result[f"innovation_acf_{str(lag).replace('.', 'p')}s"] = (
            float(innovation_acf) if np.isfinite(innovation_acf) else 0.0
        )
    return result


def feature_errors(observed: np.ndarray, simulated: np.ndarray, interval_s: float) -> dict[str, float]:
    obs = temporal_features(observed, interval_s)
    sim = temporal_features(simulated, interval_s)
    eps = 1e-6
    scale = max(obs["std_kw"], eps)
    variability = abs(math.log((sim["std_kw"] + eps) / (obs["std_kw"] + eps)))
    distribution = 0.5 * float(wasserstein_distance(observed, simulated)) / scale
    distribution += 0.5 * float(
        np.mean(
            [
                abs(sim[column] - obs[column]) / scale
                for column in ("p01_kw", "p05_kw", "p95_kw", "p99_kw")
            ]
        )
    )
    acf_columns = [f"acf_{str(lag).replace('.', 'p')}s" for lag in ACF_LAGS_S]
    temporal = float(np.sqrt(np.mean([(sim[column] - obs[column]) ** 2 for column in acf_columns])))
    innovation_acf_columns = [
        f"innovation_acf_{str(lag).replace('.', 'p')}s" for lag in ACF_LAGS_S
    ]
    innovation_temporal = float(
        np.sqrt(
            np.mean([(sim[column] - obs[column]) ** 2 for column in innovation_acf_columns])
        )
    )
    ramp = float(
        np.mean(
            [
                abs(math.log((sim[column] + eps) / (obs[column] + eps)))
                for column in ("p95_abs_ramp_kw_per_s", "p99_abs_ramp_kw_per_s")
            ]
        )
    )
    spectral = abs(sim["low_frequency_psd_share"] - obs["low_frequency_psd_share"])
    tail_ratio_error = abs(
        math.log(
            (sim["innovation_gaussian_tail_ratio"] + eps)
            / (obs["innovation_gaussian_tail_ratio"] + eps)
        )
    )
    kurtosis_error = abs(
        sim["innovation_excess_kurtosis"] - obs["innovation_excess_kurtosis"]
    ) / (abs(obs["innovation_excess_kurtosis"]) + 5.0)
    tail = 0.6 * tail_ratio_error + 0.4 * kurtosis_error
    score = (
        0.15 * variability
        + 0.15 * distribution
        + 0.20 * temporal
        + 0.15 * ramp
        + 0.10 * spectral
        + 0.15 * tail
        + 0.10 * innovation_temporal
    )
    return {
        "overall_score": score,
        "variability_error": variability,
        "distribution_error": distribution,
        "acf_error": temporal,
        "ramp_error": ramp,
        "spectral_error": spectral,
        "innovation_tail_error": tail,
        "innovation_acf_error": innovation_temporal,
    }


def _stable_seed(base_seed: int, *parts: str) -> int:
    return int((base_seed + zlib.crc32("|".join(parts).encode("utf-8"))) % (2**32 - 1))


def evaluate_models(
    *,
    profiles: Sequence[PowerProfile],
    evaluation_split: str,
    model_names: Sequence[str],
    m3_parameters: Mapping[str, Any],
    library: ResidualParameterLibrary,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    selected = [profile for profile in profiles if profile.split == evaluation_split]
    for profile in tqdm(selected, desc=f"Evaluating {evaluation_split} residual models"):
        steady = float(m3_parameters[profile.prompt_dataset].predict(profile.request_rate))
        observed = profile.power_kw - steady
        observed = observed - np.mean(observed)
        for model_name in model_names:
            repetitions = 1 if model_name == "R0_m3_only" else draws
            error_rows = []
            first: np.ndarray | None = None
            for draw in range(repetitions):
                rng = np.random.default_rng(
                    _stable_seed(seed, evaluation_split, profile.profile_id, model_name, str(draw))
                )
                simulated = simulate_residual(
                    model_name,
                    length=len(observed),
                    prompt_dataset=profile.prompt_dataset,
                    request_rate=profile.request_rate,
                    library=library,
                    rng=rng,
                )
                first = simulated if first is None else first
                error_rows.append(feature_errors(observed, simulated, profile.interval_s))
            mean_error = pd.DataFrame(error_rows).mean(numeric_only=True).to_dict()
            rows.append(
                {
                    "evaluation_split": evaluation_split,
                    "profile_id": profile.profile_id,
                    "prompt_dataset": profile.prompt_dataset,
                    "request_rate": profile.request_rate,
                    "model": model_name,
                    "draws": repetitions,
                    **mean_error,
                }
            )
            if profile.request_rate in {50.0, 90.0}:
                assert first is not None
                for index, time_s in enumerate(profile.time_s):
                    trace_rows.append(
                        {
                            "evaluation_split": evaluation_split,
                            "profile_id": profile.profile_id,
                            "prompt_dataset": profile.prompt_dataset,
                            "request_rate": profile.request_rate,
                            "model": model_name,
                            "time_s": time_s,
                            "m3_steady_power_kw": steady,
                            "measured_power_kw": profile.power_kw[index],
                            "simulated_power_kw": steady + first[index],
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(trace_rows)


def aggregate_metrics(per_profile: pd.DataFrame) -> pd.DataFrame:
    components = [column for column in per_profile if column.endswith("_error")]
    rows = []
    for model, group in per_profile.groupby("model", observed=True):
        row: dict[str, Any] = {
            "model": model,
            "profile_count": int(group["profile_id"].nunique()),
            "mean_overall_score": float(group["overall_score"].mean()),
            "median_overall_score": float(group["overall_score"].median()),
            "sem_overall_score": float(group["overall_score"].std(ddof=1) / math.sqrt(len(group))),
        }
        for column in components:
            row[f"mean_{column}"] = float(group[column].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_overall_score", kind="stable")


def choose_model(validation_metrics: pd.DataFrame) -> dict[str, Any]:
    eligible = validation_metrics.loc[
        validation_metrics["model"].isin(DEPLOYABLE_MODELS)
    ].sort_values(["mean_overall_score", "mean_acf_error", "mean_ramp_error"], kind="stable")
    selected = str(eligible.iloc[0]["model"])
    return {
        "selected_residual_model": selected,
        "selection_split": "validation",
        "eligible_models": list(DEPLOYABLE_MODELS),
        "excluded_reference": "R4_empirical_replay",
        "mad_used": False,
        "selection_rule": "minimum validation mean statistical-reconstruction score",
    }


def paired_model_comparison(
    per_profile: pd.DataFrame,
    selected_model: str,
    *,
    bootstrap_samples: int = 20_000,
    seed: int = 42,
) -> pd.DataFrame:
    selected = per_profile.loc[
        per_profile["model"].eq(selected_model), ["profile_id", "overall_score"]
    ].rename(columns={"overall_score": "selected_score"})
    rng = np.random.default_rng(seed)
    rows = []
    for competitor in DEPLOYABLE_MODELS:
        if competitor == selected_model:
            continue
        other = per_profile.loc[
            per_profile["model"].eq(competitor), ["profile_id", "overall_score"]
        ].rename(columns={"overall_score": "competitor_score"})
        paired = selected.merge(other, on="profile_id", validate="one_to_one")
        difference = paired["competitor_score"].to_numpy(float) - paired[
            "selected_score"
        ].to_numpy(float)
        indices = rng.integers(0, len(difference), size=(bootstrap_samples, len(difference)))
        bootstrap_mean = difference[indices].mean(axis=1)
        rows.append(
            {
                "selected_model": selected_model,
                "competitor_model": competitor,
                "profile_count": len(difference),
                "mean_score_advantage": float(np.mean(difference)),
                "selected_win_fraction": float(np.mean(difference > 0.0)),
                "bootstrap_ci95_low": float(np.quantile(bootstrap_mean, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(bootstrap_mean, 0.975)),
                "interpretation": "positive values favor selected model",
            }
        )
    return pd.DataFrame(rows)


def score_weight_sensitivity(
    aggregate: pd.DataFrame,
    *,
    samples: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    eligible = aggregate.loc[aggregate["model"].isin(DEPLOYABLE_MODELS)].copy()
    columns = [f"mean_{column}" for column in ERROR_COMPONENTS]
    values = eligible[columns].to_numpy(float)
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(columns)), size=samples)
    winner = np.argmin(weights @ values.T, axis=1)
    names = eligible["model"].to_numpy()[winner]
    counts = pd.Series(names).value_counts()
    return pd.DataFrame(
        {
            "model": eligible["model"],
            "selection_count": [int(counts.get(name, 0)) for name in eligible["model"]],
            "selection_fraction": [float(counts.get(name, 0) / samples) for name in eligible["model"]],
            "design": "Dirichlet(1,...,1) over seven error-component weights",
        }
    ).sort_values("selection_fraction", ascending=False, kind="stable")


def _plot_scores(metrics: pd.DataFrame, title: str, output: Path) -> None:
    ordered = metrics.sort_values("mean_overall_score", ascending=True)
    colors = ["#9ca3af" if name == "R4_empirical_replay" else "#2563eb" for name in ordered["model"]]
    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=160)
    ax.barh(ordered["model"], ordered["mean_overall_score"], color=colors)
    ax.set_xlabel("Mean reconstruction score (lower is better)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_representative_test(traces: pd.DataFrame, selected_model: str, output: Path) -> None:
    working = traces.loc[
        traces["evaluation_split"].eq("test")
        & traces["request_rate"].eq(90.0)
        & traces["model"].isin({selected_model, "R4_empirical_replay"})
    ]
    datasets = working["prompt_dataset"].drop_duplicates().tolist()
    if not datasets:
        return
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 4.0 * len(datasets)), dpi=160, sharex=True)
    axes = np.atleast_1d(axes)
    for axis, dataset in zip(axes, datasets, strict=True):
        part = working.loc[working["prompt_dataset"].eq(dataset)]
        selected = part.loc[part["model"].eq(selected_model)]
        reference = part.loc[part["model"].eq("R4_empirical_replay")]
        axis.plot(selected["time_s"], selected["measured_power_kw"], color="#111827", lw=0.9, label="measured")
        axis.plot(selected["time_s"], selected["simulated_power_kw"], color="#2563eb", lw=0.8, label=selected_model)
        if not reference.empty:
            axis.plot(reference["time_s"], reference["simulated_power_kw"], color="#f97316", lw=0.75, alpha=0.85, label="R4 empirical replay")
        axis.axhline(float(selected["m3_steady_power_kw"].iloc[0]), color="#059669", lw=1.0, label="M3 steady")
        axis.set_title(f"{dataset} | 90 request/s high-rate test")
        axis.set_ylabel("IT power (kW)")
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{float(value):.4g}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_residual_model_selection(
    *,
    dataset_root: str | Path,
    manifest_path: str | Path,
    condition_target_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    draws: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles(dataset_root=dataset_root, manifest_path=manifest_path, split_path=split_path)
    targets = load_condition_targets(condition_target_path, split_path)

    validation_m3 = fit_m3_by_dataset(targets, {"train"})
    validation_library = build_parameter_library(profiles, validation_m3, {"train"})
    validation_profiles, validation_traces = evaluate_models(
        profiles=profiles,
        evaluation_split="validation",
        model_names=MODEL_NAMES,
        m3_parameters=validation_m3,
        library=validation_library,
        draws=draws,
        seed=seed,
    )
    validation_metrics = aggregate_metrics(validation_profiles)
    decision = choose_model(validation_metrics)
    paired = paired_model_comparison(
        validation_profiles, decision["selected_residual_model"], seed=seed
    )
    weight_sensitivity = score_weight_sensitivity(validation_metrics, seed=seed)
    selected_pairing = paired.loc[
        paired["competitor_model"].eq("R3_jump_ou")
    ]
    if not selected_pairing.empty:
        comparison = selected_pairing.iloc[0]
        paired_conclusive = bool(float(comparison["bootstrap_ci95_low"]) > 0.0)
        decision.update(
            {
                "selection_status": (
                    "statistically_supported" if paired_conclusive else "provisional_statistical_baseline"
                ),
                "r2_vs_r3_paired_ci95": [
                    float(comparison["bootstrap_ci95_low"]),
                    float(comparison["bootstrap_ci95_high"]),
                ],
                "r2_vs_r3_selected_win_fraction": float(
                    comparison["selected_win_fraction"]
                ),
                "final_family_selection_pending": not paired_conclusive,
            }
        )
    selected_weight_row = weight_sensitivity.loc[
        weight_sensitivity["model"].eq(decision["selected_residual_model"])
    ]
    if not selected_weight_row.empty:
        decision["selected_under_random_weight_fraction"] = float(
            selected_weight_row.iloc[0]["selection_fraction"]
        )
    decision["downstream_note"] = (
        "Final choice must also compare DOE violation and decision-regret metrics; "
        "this stage selects only a statistical gray-box baseline."
    )
    by_dataset = (
        validation_profiles.groupby(["prompt_dataset", "model"], observed=True)[
            ["overall_score", *ERROR_COMPONENTS]
        ]
        .mean()
        .reset_index()
        .sort_values(["prompt_dataset", "overall_score"], kind="stable")
    )
    score_wide = validation_profiles.pivot_table(
        index=["profile_id", "prompt_dataset", "request_rate"],
        columns="model",
        values="overall_score",
        aggfunc="first",
    ).reset_index()
    score_wide["r3_minus_r2_score"] = score_wide["R3_jump_ou"] - score_wide["R2_ou"]

    test_m3 = fit_m3_by_dataset(targets, {"train", "validation"})
    test_library = build_parameter_library(profiles, test_m3, {"train", "validation"})
    test_models = (decision["selected_residual_model"], "R4_empirical_replay")
    test_profiles, test_traces = evaluate_models(
        profiles=profiles,
        evaluation_split="test",
        model_names=test_models,
        m3_parameters=test_m3,
        library=test_library,
        draws=draws,
        seed=seed,
    )
    test_metrics = aggregate_metrics(test_profiles)

    validation_library.parameters.to_csv(output / "train_residual_parameters.csv", index=False)
    test_library.parameters.to_csv(output / "train_validation_residual_parameters.csv", index=False)
    validation_profiles.to_csv(output / "validation_per_profile_metrics.csv", index=False)
    validation_metrics.to_csv(output / "validation_aggregate_metrics.csv", index=False)
    by_dataset.to_csv(output / "validation_metrics_by_prompt_dataset.csv", index=False)
    paired.to_csv(output / "validation_paired_model_comparison.csv", index=False)
    weight_sensitivity.to_csv(output / "validation_score_weight_sensitivity.csv", index=False)
    score_wide.to_csv(output / "validation_r3_minus_r2_by_condition.csv", index=False)
    test_profiles.to_csv(output / "test_per_profile_metrics.csv", index=False)
    test_metrics.to_csv(output / "test_aggregate_metrics.csv", index=False)
    traces = pd.concat([validation_traces, test_traces], ignore_index=True)
    traces.to_csv(output / "representative_model_traces.csv", index=False)
    (output / "selected_residual_model.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot_scores(validation_metrics, "Validation residual-model comparison", output / "validation_model_scores.png")
    _plot_scores(test_metrics, "High-rate test result", output / "test_model_scores.png")
    _plot_representative_test(traces, decision["selected_residual_model"], output / "representative_high_rate_test.png")

    report_columns = [
        "model",
        "profile_count",
        "mean_overall_score",
        "mean_acf_error",
        "mean_ramp_error",
        "mean_innovation_tail_error",
    ]
    report = f"""# MAD-free residual 모델 비교 결과

## 구조

- R1: Gaussian white noise baseline
- R2: OU/AR(1) with Gaussian innovation
- R3: signed Jump-OU의 이산시간 근사(negative/background/positive innovation mixture)
- R3의 jump intensity는 train condition의 BIC·tail 진단에 따라 0까지 축소될 수 있도록 보정
- R4: train residual empirical replay; 비모수 reference이므로 선택 대상에서 제외
- MAD event threshold는 사용하지 않음

## Validation 결과

{_markdown_table(validation_metrics[report_columns])}

## 선택

```json
{json.dumps(decision, indent=2, ensure_ascii=False)}
```

### 선택모델의 paired 비교

{_markdown_table(paired)}

### Score 가중치 민감도

{_markdown_table(weight_sensitivity)}

## High-rate test 결과

{_markdown_table(test_metrics[report_columns])}

## 주의사항

- R3의 GMM은 small-dt에서 한 sample에 최대 한 개의 signed jump가 발생한다고 본 Jump-OU transition mixture 근사다.
- `selection_status=provisional_statistical_baseline`이면 R2/R3의 paired 차이가 확정적이지 않으므로 최종 family로 고정하면 안 된다.
- 최종 선택은 pointwise waveform 일치가 아니라 분산, 분위수, ACF, ramp, PSD, innovation tail 재현 score를 기준으로 한다.
- DFL 단계에서는 DOE 위반, constraint violation 및 decision regret으로 R2/R3를 다시 비교한다.
- test 단계에서는 선택된 family를 train+validation에서 다시 보정한 뒤 81--100 request/s에 적용했다.
"""
    (output / "residual_model_selection_report_ko.md").write_text(report, encoding="utf-8")
    return {
        "output_dir": output,
        "decision": decision,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
