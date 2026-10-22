"""Audit whether sustained NLR online-inference traces identify IT-load dynamics.

The NLR online-rate experiment records output power under a constant request-rate
condition.  It does not provide a time-varying request-rate input channel.  This
module therefore distinguishes:

* an *apparent output transient* fitted at the beginning of a trace, and
* a causal input-output time constant, which is not identifiable without a
  documented input transition.

The selected prompt-conditioned M3 logistic map supplies the steady-state target.
Temporal residuals are then characterised separately for later gray-box work.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.signal import welch
from tqdm.auto import tqdm

from .readers import read_timeseries
from .signal_validation import apply_analysis_policy


WORKLOAD_TYPE = "online_inference_rate"
REPRESENTATIVE_RATES = (1.0, 10.0, 25.0, 50.0, 100.0)


@dataclass(frozen=True)
class M3Parameters:
    p0_kw: float
    delta_p_kw: float
    k: float
    x0: float

    @property
    def saturation_power_kw(self) -> float:
        return self.p0_kw + self.delta_p_kw

    @property
    def half_rate_request_s(self) -> float:
        return float(np.exp(self.x0) - 1.0)

    def predict(self, request_rate: float | np.ndarray) -> np.ndarray:
        rate = np.asarray(request_rate, dtype=float)
        x = np.log1p(np.maximum(rate, 0.0))
        argument = np.clip(-self.k * (x - self.x0), -700.0, 700.0)
        return self.p0_kw + self.delta_p_kw / (1.0 + np.exp(argument))


def _prompt_dataset(value: Any) -> str:
    return str(value).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_m3_parameters(path: str | Path) -> dict[str, M3Parameters]:
    table = pd.read_csv(Path(path).expanduser().resolve())
    selected = table.loc[table["model"].eq("M3_logistic")].copy()
    required = {"prompt_dataset", "parameter", "value"}
    missing = required.difference(selected.columns)
    if missing:
        raise ValueError(f"M3 parameter table is missing columns: {sorted(missing)}")
    result: dict[str, M3Parameters] = {}
    for dataset, group in selected.groupby("prompt_dataset", observed=True):
        values = group.set_index("parameter")["value"].to_dict()
        expected = {"p0", "delta_p", "k", "x0"}
        if not expected.issubset(values):
            raise ValueError(f"Incomplete M3 parameters for {dataset}: {values}")
        result[str(dataset)] = M3Parameters(
            p0_kw=float(values["p0"]),
            delta_p_kw=float(values["delta_p"]),
            k=float(values["k"]),
            x0=float(values["x0"]),
        )
    if not result:
        raise ValueError("No M3 parameters found")
    return result


def select_manifest_profiles(manifest_path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(Path(manifest_path).expanduser().resolve())
    required = {
        "profile_id",
        "relative_path",
        "workload_type",
        "analysis_cohort",
        "eligible",
        "analysis_interval_s",
        "resampling_method",
        "request_rate",
        "dataset_path",
        "measurement_origin",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Audited manifest is missing columns: {sorted(missing)}")
    selected = manifest.loc[
        manifest["workload_type"].eq(WORKLOAD_TYPE)
        & manifest["analysis_cohort"].eq("measured_it_workload")
        & _as_bool(manifest["eligible"])
    ].copy()
    selected["request_rate"] = pd.to_numeric(selected["request_rate"], errors="coerce")
    selected["analysis_interval_s"] = pd.to_numeric(
        selected["analysis_interval_s"], errors="coerce"
    )
    selected["prompt_dataset"] = selected["dataset_path"].map(_prompt_dataset)
    selected = selected.dropna(subset=["request_rate", "analysis_interval_s"])
    if selected.empty:
        raise ValueError("No eligible online-inference-rate profiles found")
    duplicate = selected.duplicated(["prompt_dataset", "request_rate"], keep=False)
    if duplicate.any():
        cases = selected.loc[duplicate, ["prompt_dataset", "request_rate"]]
        raise ValueError(f"Duplicate prompt/rate profiles found:\n{cases.to_string(index=False)}")
    return selected.sort_values(["prompt_dataset", "request_rate"], kind="stable")


def _finite_signal(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    time_s = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(float)
    power_kw = pd.to_numeric(frame["power_w"], errors="coerce").to_numpy(float) / 1000.0
    valid = np.isfinite(time_s) & np.isfinite(power_kw)
    time_s, power_kw = time_s[valid], power_kw[valid]
    order = np.argsort(time_s, kind="stable")
    time_s, power_kw = time_s[order], power_kw[order]
    if len(time_s) < 20:
        raise ValueError("Profile has fewer than 20 finite samples")
    time_s = time_s - time_s[0]
    positive = np.diff(time_s)
    positive = positive[positive > 0]
    if positive.size == 0:
        raise ValueError("Profile has no positive time interval")
    return time_s, power_kw, float(np.median(positive))


def _autocorrelation(values: np.ndarray, lag_samples: int) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if lag_samples <= 0:
        return 1.0
    if len(x) <= lag_samples + 2:
        return float("nan")
    x = x - np.mean(x)
    denominator = float(np.dot(x, x))
    if denominator <= np.finfo(float).eps:
        return float("nan")
    return float(np.dot(x[:-lag_samples], x[lag_samples:]) / denominator)


def _event_statistics(
    values: np.ndarray,
    threshold: float,
    *,
    direction: str,
    interval_s: float,
) -> dict[str, float | int]:
    if direction == "above":
        mask = values > threshold
        exceedance = values - threshold
    elif direction == "below":
        mask = values < threshold
        exceedance = threshold - values
    else:
        raise ValueError(direction)
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = changes[::2], changes[1::2]
    lengths = stops - starts
    peaks = [float(np.max(exceedance[start:stop])) for start, stop in zip(starts, stops)]
    prefix = "burst" if direction == "above" else "dip"
    return {
        f"{prefix}_count": int(len(lengths)),
        f"{prefix}_fraction": float(np.mean(mask)),
        f"{prefix}_mean_duration_s": float(np.mean(lengths) * interval_s) if len(lengths) else 0.0,
        f"{prefix}_max_duration_s": float(np.max(lengths) * interval_s) if len(lengths) else 0.0,
        f"{prefix}_max_exceedance_kw": float(np.max(peaks)) if peaks else 0.0,
    }


def _apparent_startup_fit(
    time_s: np.ndarray,
    power_kw: np.ndarray,
    *,
    interval_s: float,
    initial_window_s: float = 1.0,
    tail_window_s: float = 30.0,
    fit_window_s: float = 30.0,
) -> dict[str, Any]:
    initial = power_kw[time_s <= initial_window_s]
    tail = power_kw[time_s >= max(time_s[-1] - tail_window_s, 0.0)]
    segment_mask = time_s <= min(fit_window_s, time_s[-1])
    t = time_s[segment_mask]
    y = power_kw[segment_mask]
    p_initial = float(np.median(initial))
    p_tail = float(np.median(tail))
    amplitude = p_tail - p_initial
    tail_median = float(np.median(tail))
    tail_mad = float(np.median(np.abs(tail - tail_median)))
    robust_tail_sigma = 1.4826 * tail_mad
    if robust_tail_sigma <= np.finfo(float).eps:
        robust_tail_sigma = float(np.std(tail, ddof=0))
    amplitude_snr = abs(amplitude) / max(robust_tail_sigma, 1e-9)
    relative_change = abs(amplitude) / max(abs(p_tail), 1e-9)

    def prediction(tau: float) -> np.ndarray:
        return p_tail + (p_initial - p_tail) * np.exp(-t / tau)

    lower = max(interval_s, 1e-3)
    upper = max(min(fit_window_s * 5.0, max(time_s[-1], lower * 2)), lower * 1.01)
    optimum = minimize_scalar(
        lambda tau: float(np.mean(np.square(prediction(tau) - y))),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-6},
    )
    tau = float(optimum.x)
    fitted = prediction(tau)
    sse = float(np.sum(np.square(y - fitted)))
    variance = float(np.sum(np.square(y - np.mean(y))))
    r2 = 1.0 - sse / variance if variance > np.finfo(float).eps else float("nan")

    smooth_samples = max(1, int(round(1.0 / interval_s)))
    smoothed = pd.Series(y).rolling(smooth_samples, center=True, min_periods=1).mean().to_numpy()
    differences = np.diff(smoothed)
    desired_sign = np.sign(amplitude)
    directional_fraction = (
        float(np.mean(differences * desired_sign >= 0))
        if desired_sign != 0 and differences.size
        else 0.0
    )

    target = p_initial + 0.6321205588 * amplitude
    if amplitude > 0:
        crossings = np.flatnonzero(y >= target)
    elif amplitude < 0:
        crossings = np.flatnonzero(y <= target)
    else:
        crossings = np.array([], dtype=int)
    crossing_time = float(t[crossings[0]]) if crossings.size else float("nan")

    candidate = bool(
        relative_change >= 0.01
        and amplitude_snr >= 3.0
        and np.isfinite(r2)
        and r2 >= 0.50
        and directional_fraction >= 0.55
        and lower * 1.01 < tau < upper * 0.99
    )
    return {
        "initial_power_kw": p_initial,
        "tail_power_kw": p_tail,
        "initial_to_tail_change_kw": amplitude,
        "initial_to_tail_change_pct": 100.0 * amplitude / max(abs(p_tail), 1e-9),
        "tail_robust_sigma_kw": robust_tail_sigma,
        "startup_change_snr": amplitude_snr,
        "apparent_tau_fit_s": tau,
        "apparent_tau_63pct_s": crossing_time,
        "apparent_exponential_r2": r2,
        "startup_directional_fraction": directional_fraction,
        "startup_transient_candidate": candidate,
        "causal_tau_identifiable": False,
        "tau_identifiability_status": (
            "apparent_output_transient_only_no_input_signal"
            if candidate
            else "not_estimable_no_input_transition"
        ),
    }


def _residual_features(
    time_s: np.ndarray,
    power_kw: np.ndarray,
    steady_power_kw: float,
    interval_s: float,
) -> tuple[dict[str, Any], np.ndarray]:
    residual = power_kw - steady_power_kw
    centered = residual - np.mean(residual)
    median = float(np.median(centered))
    mad = float(np.median(np.abs(centered - median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= np.finfo(float).eps:
        robust_sigma = float(np.std(centered, ddof=0))

    ramp = np.diff(power_kw) / np.diff(time_s)
    finite_ramp = ramp[np.isfinite(ramp)]
    lag_01 = max(1, int(round(0.1 / interval_s)))
    lag_1 = max(1, int(round(1.0 / interval_s)))
    lag_10 = max(1, int(round(10.0 / interval_s)))

    if len(centered) >= 4 and np.dot(centered[:-1], centered[:-1]) > np.finfo(float).eps:
        phi = float(np.dot(centered[:-1], centered[1:]) / np.dot(centered[:-1], centered[:-1]))
    else:
        phi = float("nan")
    correlation_time = (
        float(-interval_s / np.log(phi)) if np.isfinite(phi) and 0.0 < phi < 1.0 else float("nan")
    )

    sampling_hz = 1.0 / interval_s
    frequencies, spectrum = welch(
        centered,
        fs=sampling_hz,
        nperseg=min(1024, len(centered)),
        detrend="linear",
    )
    positive = frequencies > 0
    if positive.any() and np.sum(spectrum[positive]) > 0:
        positive_frequency = frequencies[positive]
        positive_spectrum = spectrum[positive]
        dominant_frequency = float(positive_frequency[np.argmax(positive_spectrum)])
        dominant_period = 1.0 / dominant_frequency if dominant_frequency > 0 else float("nan")
        low_frequency_share = float(
            np.sum(positive_spectrum[positive_frequency <= 0.1]) / np.sum(positive_spectrum)
        )
    else:
        dominant_period = float("nan")
        low_frequency_share = float("nan")

    slope_kw_s = float(np.polyfit(time_s - np.mean(time_s), power_kw, 1)[0])
    window = min(30.0, time_s[-1] / 4.0)
    first_mean = float(np.mean(power_kw[time_s <= window]))
    last_mean = float(np.mean(power_kw[time_s >= time_s[-1] - window]))
    stationarity_shift = last_mean - first_mean

    upper = median + 3.0 * max(robust_sigma, 1e-9)
    lower = median - 3.0 * max(robust_sigma, 1e-9)
    burst = _event_statistics(centered, upper, direction="above", interval_s=interval_s)
    dip = _event_statistics(centered, lower, direction="below", interval_s=interval_s)
    acf_1s = _autocorrelation(centered, lag_1)
    event_count = int(burst["burst_count"]) + int(dip["dip_count"])
    if np.isfinite(acf_1s) and abs(acf_1s) >= 0.20 and event_count:
        residual_candidate = "correlated_residual_plus_events"
    elif np.isfinite(acf_1s) and abs(acf_1s) >= 0.20:
        residual_candidate = "ar1_or_ou_residual"
    elif event_count:
        residual_candidate = "weakly_correlated_noise_plus_events"
    else:
        residual_candidate = "approximately_white_residual"

    features = {
        "m3_steady_power_kw": steady_power_kw,
        "profile_mean_power_kw": float(np.mean(power_kw)),
        "m3_bias_kw": float(np.mean(residual)),
        "m3_profile_rmse_kw": float(np.sqrt(np.mean(np.square(residual)))),
        "residual_std_kw": float(np.std(centered, ddof=0)),
        "residual_robust_sigma_kw": robust_sigma,
        "residual_p05_kw": float(np.quantile(centered, 0.05)),
        "residual_p95_kw": float(np.quantile(centered, 0.95)),
        "residual_acf_0p1s": _autocorrelation(centered, lag_01),
        "residual_acf_1s": acf_1s,
        "residual_acf_10s": _autocorrelation(centered, lag_10),
        "residual_ar1_phi": phi,
        "residual_correlation_time_s": correlation_time,
        "residual_dominant_period_s": dominant_period,
        "residual_low_frequency_psd_share": low_frequency_share,
        "p95_abs_ramp_kw_per_s": float(np.quantile(np.abs(finite_ramp), 0.95)),
        "max_ramp_up_kw_per_s": float(np.max(finite_ramp)),
        "max_ramp_down_kw_per_s": float(np.min(finite_ramp)),
        "linear_trend_kw_per_min": slope_kw_s * 60.0,
        "first_last_30s_shift_kw": stationarity_shift,
        "first_last_shift_in_sigma": stationarity_shift / max(robust_sigma, 1e-9),
        **burst,
        **dip,
        "residual_model_candidate": residual_candidate,
    }
    return features, centered


def analyze_profile(
    frame: pd.DataFrame,
    record: Mapping[str, Any],
    parameters: M3Parameters,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    time_s, power_kw, interval_s = _finite_signal(frame)
    request_rate = float(record["request_rate"])
    steady_power = float(parameters.predict(request_rate))
    startup = _apparent_startup_fit(
        time_s, power_kw, interval_s=interval_s
    )
    residual, centered = _residual_features(
        time_s, power_kw, steady_power, interval_s
    )
    common = {
        "profile_id": record["profile_id"],
        "relative_path": record["relative_path"],
        "prompt_dataset": record["prompt_dataset"],
        "request_rate": request_rate,
        "measurement_origin": record["measurement_origin"],
        "sample_count": len(time_s),
        "duration_s": float(time_s[-1]),
        "analysis_interval_s": interval_s,
        "input_transition_signal_available": False,
    }
    identifiability = {**common, **startup}
    residual_row = {**common, **residual}

    trace = pd.DataFrame(
        {
            **{key: value for key, value in common.items() if key not in {"sample_count", "duration_s"}},
            "time_s": time_s,
            "measured_power_kw": power_kw,
            "m3_steady_power_kw": steady_power,
            "raw_residual_kw": power_kw - steady_power,
            "centered_residual_kw": centered,
        }
    )
    max_lag_s = min(20.0, float(time_s[-1]) / 4.0)
    lag_seconds = np.arange(0.0, max_lag_s + interval_s / 2.0, interval_s)
    acf = pd.DataFrame(
        {
            "profile_id": record["profile_id"],
            "prompt_dataset": record["prompt_dataset"],
            "request_rate": request_rate,
            "lag_s": lag_seconds,
            "autocorrelation": [
                _autocorrelation(centered, int(round(lag / interval_s)))
                for lag in lag_seconds
            ],
        }
    )
    return identifiability, residual_row, trace, acf


def _summary_tables(
    identifiability: pd.DataFrame,
    residual: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for dataset, group in residual.groupby("prompt_dataset", observed=True):
        ident = identifiability.loc[identifiability["prompt_dataset"].eq(dataset)]
        rows.append(
            {
                "scope": f"dataset:{dataset}",
                "profile_count": len(group),
                "apparent_startup_candidate_count": int(ident["startup_transient_candidate"].sum()),
                "causal_tau_identifiable_count": int(ident["causal_tau_identifiable"].sum()),
                "median_m3_bias_kw": float(group["m3_bias_kw"].median()),
                "median_residual_std_kw": float(group["residual_std_kw"].median()),
                "median_residual_acf_1s": float(group["residual_acf_1s"].median()),
                "median_residual_acf_10s": float(group["residual_acf_10s"].median()),
                "median_residual_correlation_time_s": float(group["residual_correlation_time_s"].median()),
                "median_p95_abs_ramp_kw_per_s": float(group["p95_abs_ramp_kw_per_s"].median()),
                "profiles_with_burst_or_dip": int(((group["burst_count"] + group["dip_count"]) > 0).sum()),
            }
        )
    overall_ident = identifiability
    rows.append(
        {
            "scope": "overall",
            "profile_count": len(residual),
            "apparent_startup_candidate_count": int(overall_ident["startup_transient_candidate"].sum()),
            "causal_tau_identifiable_count": int(overall_ident["causal_tau_identifiable"].sum()),
            "median_m3_bias_kw": float(residual["m3_bias_kw"].median()),
            "median_residual_std_kw": float(residual["residual_std_kw"].median()),
            "median_residual_acf_1s": float(residual["residual_acf_1s"].median()),
            "median_residual_acf_10s": float(residual["residual_acf_10s"].median()),
            "median_residual_correlation_time_s": float(residual["residual_correlation_time_s"].median()),
            "median_p95_abs_ramp_kw_per_s": float(residual["p95_abs_ramp_kw_per_s"].median()),
            "profiles_with_burst_or_dip": int(((residual["burst_count"] + residual["dip_count"]) > 0).sum()),
        }
    )
    summary = pd.DataFrame(rows)
    candidate_counts = residual["residual_model_candidate"].value_counts().to_dict()
    decision = {
        "causal_first_order_tau_identifiable": False,
        "reason": (
            "The sustained-rate files contain output power only and no time-varying "
            "request-rate input or documented step timestamp. Apparent startup fits "
            "cannot be interpreted as causal plant time constants."
        ),
        "recommended_dynamic_representation_with_current_data": (
            "prompt-conditioned M3 steady-state backbone plus condition-dependent "
            "correlated residual/event model or empirical residual replay"
        ),
        "data_required_for_physical_tau": (
            "synchronized request-rate step/ramp command and IT power response, "
            "including pre-step baseline and post-step settling"
        ),
        "apparent_startup_candidate_count": int(identifiability["startup_transient_candidate"].sum()),
        "profile_count": int(len(identifiability)),
        "residual_model_candidate_counts": {str(k): int(v) for k, v in candidate_counts.items()},
    }
    return summary, decision


def _plot_representative_profiles(trace: pd.DataFrame, output: Path) -> None:
    datasets = list(trace["prompt_dataset"].drop_duplicates())
    rates = sorted(trace["request_rate"].unique())
    fig, axes = plt.subplots(len(rates), len(datasets), figsize=(7 * len(datasets), 2.5 * len(rates)), dpi=150, sharex=True)
    axes = np.asarray(axes).reshape(len(rates), len(datasets))
    for row, rate in enumerate(rates):
        for column, dataset in enumerate(datasets):
            ax = axes[row, column]
            part = trace.loc[
                trace["request_rate"].eq(rate) & trace["prompt_dataset"].eq(dataset)
            ]
            if part.empty:
                ax.set_visible(False)
                continue
            ax.plot(part["time_s"], part["measured_power_kw"], lw=0.9, color="#2563eb", label="measured")
            ax.axhline(part["m3_steady_power_kw"].iloc[0], color="#059669", lw=2.0, label="M3 steady target")
            ax.set_title(f"{dataset} | {rate:g} request/s", fontsize=9)
            ax.grid(alpha=0.22)
            if column == 0:
                ax.set_ylabel("IT power (kW)")
            if row == len(rates) - 1:
                ax.set_xlabel("Time (s)")
    axes[0, -1].legend(fontsize=8)
    fig.suptitle("Representative sustained-rate NLR profiles vs M3 steady-state target")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_residual_diagnostics(residual: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), dpi=150, sharex=True)
    colors = ["#2563eb", "#f97316", "#059669", "#7c3aed"]
    for color, (dataset, group) in zip(colors, residual.groupby("prompt_dataset", observed=True)):
        ordered = group.sort_values("request_rate")
        axes[0].plot(ordered["request_rate"], ordered["m3_bias_kw"], marker="o", ms=3, lw=1.2, color=color, label=dataset)
        axes[1].plot(ordered["request_rate"], ordered["residual_std_kw"], marker="o", ms=3, lw=1.2, color=color, label=dataset)
        axes[2].plot(ordered["request_rate"], ordered["p95_abs_ramp_kw_per_s"], marker="o", ms=3, lw=1.2, color=color, label=dataset)
    axes[0].axhline(0, color="#111827", lw=0.9)
    axes[0].set_ylabel("M3 bias (kW)")
    axes[1].set_ylabel("Centered residual std (kW)")
    axes[2].set_ylabel("P95 |ramp| (kW/s)")
    axes[2].set_xlabel("Request rate (request/s)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Condition dependence of M3 residual dynamics")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_acf(acf: pd.DataFrame, output: Path) -> None:
    working = acf.copy()
    working["rate_band"] = pd.cut(
        working["request_rate"], bins=[0, 10, 25, 50, 80, np.inf],
        labels=["1-10", "11-25", "26-50", "51-80", "81-100"],
    )
    grouped = (
        working.groupby(["prompt_dataset", "rate_band", "lag_s"], observed=True)["autocorrelation"]
        .median()
        .reset_index()
    )
    datasets = list(grouped["prompt_dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), dpi=150, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets, strict=True):
        for band, part in grouped.loc[grouped["prompt_dataset"].eq(dataset)].groupby("rate_band", observed=True):
            ax.plot(part["lag_s"], part["autocorrelation"], lw=1.5, label=str(band))
        ax.axhline(0, color="#111827", lw=0.8)
        ax.set_title(str(dataset))
        ax.set_xlabel("Lag (s)")
        ax.grid(alpha=0.25)
        ax.legend(title="request/s", fontsize=8)
    axes[0].set_ylabel("Median residual autocorrelation")
    fig.suptitle("Residual autocorrelation by request-rate band")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(
    summary: pd.DataFrame,
    decision: Mapping[str, Any],
    residual: pd.DataFrame,
) -> str:
    overall = summary.loc[summary["scope"].eq("overall")].iloc[0]
    counts = residual["residual_model_candidate"].value_counts()
    count_lines = "\n".join(f"- {name}: {int(count)} profiles" for name, count in counts.items())
    rounded = summary.round(5).astype(str)
    header = "| " + " | ".join(rounded.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(rounded.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in rounded.to_numpy()]
    table = "\n".join([header, rule, *rows])
    return f"""# NLR 동특성 식별 가능성 및 M3 residual 분석

## 결론

- **현재 NLR sustained-rate profile만으로 causal 1차 지연 시정수 tau를 식별할 수 없음.**
- 이유: request rate는 condition metadata로만 존재하고, profile 내부의 시간변화 입력이나 step 시점이 기록되어 있지 않음.
- 출력 시작부의 지수응답 적합은 `apparent_tau`로만 저장했으며 물리적 IT-load tau로 사용하지 않음.
- 현재 데이터에 적합한 다음 모델은 `prompt-conditioned M3 + condition-dependent residual/event model`임.

## 분석 범위

- Profile: {int(overall['profile_count'])}
- Apparent startup transient 후보: {int(overall['apparent_startup_candidate_count'])}
- Causal tau 식별 가능 profile: {int(overall['causal_tau_identifiable_count'])}
- 대표 분석해상도: 0.1 s

## Dataset별 요약

{table}

## Residual 모델 후보 분류

{count_lines}

## 해석상 주의

- `residual_correlation_time_s`는 출력 residual의 통계적 상관시간이며 입력-출력 plant time constant가 아님.
- 원자료는 `interpolated_from_measured`이므로 매우 짧은 시간대의 상관성과 ramp는 측정·보간과정의 영향을 받을 수 있음.
- M3 bias와 시간 residual을 분리함: condition 평균 오차는 calibration 문제이고, centered residual은 동적 변동 문제임.
- Burst/dip은 centered residual의 median ± 3 MAD 기준으로 탐지한 **후보 event**이며, 원인까지 확인된 물리 event 수가 아님.

## 물리적 tau 추정에 필요한 추가 데이터

1. 동기화된 request-rate command time series
2. Step 이전의 안정된 baseline
3. Step/ramp 인가시각
4. Step 이후 충분한 settling 구간
5. 동일 조건 반복실험

## 다음 구현 단계

1. 현재 residual이 시간상관성과 event를 보이면 AR(1)/OU + event 후보를 비교함.
2. Residual의 ACF, PSD, ramp, dip 지속시간을 재현하는지 condition holdout으로 검증함.
3. 별도의 controlled step 데이터가 확보되면 그때 M3 뒤에 1차 지연 block을 식별함.
"""


def run_dynamic_identifiability(
    *,
    dataset_root: str | Path,
    manifest_path: str | Path,
    parameter_path: str | Path,
    output_dir: str | Path,
    representative_rates: Iterable[float] = REPRESENTATIVE_RATES,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    manifest = select_manifest_profiles(manifest_path)
    parameters = load_m3_parameters(parameter_path)
    representative = set(float(rate) for rate in representative_rates)

    identifiability_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    representative_traces: list[pd.DataFrame] = []
    acf_frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []

    for _, record in tqdm(
        manifest.iterrows(), total=len(manifest), desc="Dynamic identifiability"
    ):
        try:
            dataset = str(record["prompt_dataset"])
            if dataset not in parameters:
                raise KeyError(f"No M3 parameters for {dataset}")
            path = root / str(record["relative_path"])
            frame = read_timeseries(
                path,
                dataset_root=root,
                include_metadata_columns=False,
                validate=True,
            )
            frame = apply_analysis_policy(frame, record)
            ident, residual, trace, acf = analyze_profile(
                frame, record, parameters[dataset]
            )
            identifiability_rows.append(ident)
            residual_rows.append(residual)
            acf_frames.append(acf)
            if float(record["request_rate"]) in representative:
                representative_traces.append(trace)
        except Exception as exc:  # record failure without silently dropping it
            failures.append(
                {
                    "profile_id": record.get("profile_id"),
                    "relative_path": record.get("relative_path"),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    identifiability = pd.DataFrame(identifiability_rows)
    residual = pd.DataFrame(residual_rows)
    if identifiability.empty or residual.empty:
        raise RuntimeError(f"No profiles were analysed; failures={failures[:3]}")
    trace = pd.concat(representative_traces, ignore_index=True)
    acf = pd.concat(acf_frames, ignore_index=True)
    summary, decision = _summary_tables(identifiability, residual)
    tau = identifiability.loc[
        identifiability["startup_transient_candidate"],
        [
            "profile_id", "prompt_dataset", "request_rate",
            "apparent_tau_fit_s", "apparent_tau_63pct_s",
            "apparent_exponential_r2", "startup_change_snr",
            "causal_tau_identifiable", "tau_identifiability_status",
        ],
    ].copy()

    tables = {
        "dynamic_identifiability_report": identifiability,
        "tau_estimates_apparent_only": tau,
        "residual_features": residual,
        "residual_acf": acf,
        "representative_traces": trace,
        "dynamic_summary": summary,
        "analysis_failures": pd.DataFrame(
            failures,
            columns=["profile_id", "relative_path", "error_type", "error_message"],
        ),
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
        if not table.empty:
            table.to_parquet(output / f"{name}.parquet", index=False)
    (output / "methodology_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "dynamic_identifiability_report_ko.md").write_text(
        _markdown_report(summary, decision, residual), encoding="utf-8"
    )
    _plot_representative_profiles(trace, figures / "representative_profiles_vs_m3.png")
    _plot_residual_diagnostics(residual, figures / "residual_diagnostics_by_request_rate.png")
    _plot_acf(acf, figures / "residual_acf_by_rate_band.png")
    return {
        "tables": tables,
        "decision": decision,
        "output_dir": output,
    }
