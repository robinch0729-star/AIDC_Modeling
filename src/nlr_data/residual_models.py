"""Compare condition-dependent stochastic residual models for NLR IT power.

The deterministic backbone is the prompt-conditioned M3 logistic map.  This
module compares temporal residual generators without treating output
autocorrelation as a causal plant time constant:

* R0: M3 only (zero residual)
* R1: condition-dependent Gaussian white noise
* R2: condition-dependent AR(1), the discrete-time OU analogue
* R3: condition-dependent AR(1) plus intermittent empirical event templates
* R4: nearest-condition empirical residual replay (reference only)

Model-family selection uses interpolation validation request rates.  The
highest request rates remain an extrapolation-only test set.  Every residual
parameter and donor trace used for one evaluation split comes only from its
calibration split.
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
from scipy.signal import welch
from scipy.stats import wasserstein_distance
from tqdm.auto import tqdm

from .dynamic_identifiability import (
    M3Parameters,
    _autocorrelation,
    _finite_signal,
    select_manifest_profiles,
)
from .readers import read_timeseries
from .signal_validation import apply_analysis_policy
from .steady_state_models import fit_model


MODEL_NAMES = (
    "R0_m3_only",
    "R1_white_noise",
    "R2_ar1",
    "R3_ar1_event_mad3",
    "R3_ar1_event_mad4",
    "R3_ar1_event_mad5",
    "R4_empirical_replay",
)
DEPLOYABLE_MODELS = MODEL_NAMES[:-1]
EVENT_THRESHOLDS = (3.0, 4.0, 5.0)
ACF_LAGS_S = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)


@dataclass(frozen=True)
class PowerProfile:
    profile_id: str
    condition_id: str
    prompt_dataset: str
    request_rate: float
    split: str
    time_s: np.ndarray
    power_kw: np.ndarray
    interval_s: float
    measurement_origin: str


@dataclass(frozen=True)
class EventTemplate:
    profile_id: str
    prompt_dataset: str
    request_rate: float
    threshold_mad: float
    sign: str
    waveform_kw: np.ndarray
    duration_s: float
    peak_kw: float


@dataclass
class ResidualLibrary:
    statistics: pd.DataFrame
    residuals: dict[str, np.ndarray]
    events: list[EventTemplate]
    profiles: dict[str, PowerProfile]

    def neighbours(
        self, prompt_dataset: str, request_rate: float, *, k: int = 8
    ) -> tuple[pd.DataFrame, np.ndarray]:
        part = self.statistics.loc[
            self.statistics["prompt_dataset"].eq(prompt_dataset)
            & self.statistics["event_threshold_mad"].eq(3.0)
        ].copy()
        if part.empty:
            raise KeyError(f"No calibration residuals for {prompt_dataset}")
        part["distance"] = np.abs(part["request_rate"] - request_rate)
        part = part.sort_values(["distance", "request_rate"], kind="stable").head(k)
        distance = part["distance"].to_numpy(float)
        weights = 1.0 / np.maximum(distance, 0.5)
        weights = weights / weights.sum()
        return part, weights

    def weighted_parameter(
        self,
        prompt_dataset: str,
        request_rate: float,
        column: str,
        *,
        threshold_mad: float = 3.0,
        k: int = 8,
    ) -> float:
        part = self.statistics.loc[
            self.statistics["prompt_dataset"].eq(prompt_dataset)
            & self.statistics["event_threshold_mad"].eq(float(threshold_mad))
        ].copy()
        if part.empty:
            raise KeyError(f"No calibration residuals for {prompt_dataset}")
        part["distance"] = np.abs(part["request_rate"] - request_rate)
        part = part.sort_values(["distance", "request_rate"], kind="stable").head(k)
        values = pd.to_numeric(part[column], errors="coerce").to_numpy(float)
        distances = part["distance"].to_numpy(float)
        valid = np.isfinite(values)
        if not valid.any():
            return float("nan")
        values, distances = values[valid], distances[valid]
        weights = 1.0 / np.maximum(distances, 0.5)
        return float(np.average(values, weights=weights))

    def neighbour_ids(
        self, prompt_dataset: str, request_rate: float, *, k: int = 8
    ) -> tuple[list[str], np.ndarray]:
        part, weights = self.neighbours(prompt_dataset, request_rate, k=k)
        return part["profile_id"].astype(str).tolist(), weights


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_profiles(
    *,
    dataset_root: str | Path,
    manifest_path: str | Path,
    split_path: str | Path,
) -> list[PowerProfile]:
    root = Path(dataset_root).expanduser().resolve()
    manifest = select_manifest_profiles(manifest_path)
    if "recommended_condition_id" not in manifest:
        raise ValueError("Manifest lacks recommended_condition_id")
    assignment = pd.read_csv(Path(split_path).expanduser().resolve())
    required = {"condition_id", "split"}
    missing = required.difference(assignment.columns)
    if missing:
        raise ValueError(f"Split assignment missing columns: {sorted(missing)}")
    assignment = assignment[["condition_id", "split"]].drop_duplicates()
    selected = manifest.merge(
        assignment,
        left_on="recommended_condition_id",
        right_on="condition_id",
        how="left",
        validate="one_to_one",
    )
    if selected["split"].isna().any():
        missing_ids = selected.loc[selected["split"].isna(), "recommended_condition_id"]
        raise ValueError(f"Profiles without split assignment: {missing_ids.head().tolist()}")

    profiles: list[PowerProfile] = []
    for _, record in tqdm(
        selected.iterrows(), total=len(selected), desc="Loading residual profiles"
    ):
        path = root / str(record["relative_path"])
        frame = read_timeseries(
            path, dataset_root=root, include_metadata_columns=False, validate=True
        )
        frame = apply_analysis_policy(frame, record)
        time_s, power_kw, interval_s = _finite_signal(frame)
        profiles.append(
            PowerProfile(
                profile_id=str(record["profile_id"]),
                condition_id=str(record["recommended_condition_id"]),
                prompt_dataset=str(record["prompt_dataset"]),
                request_rate=float(record["request_rate"]),
                split=str(record["split"]),
                time_s=time_s,
                power_kw=power_kw,
                interval_s=interval_s,
                measurement_origin=str(record["measurement_origin"]),
            )
        )
    return profiles


def load_condition_targets(
    target_path: str | Path, split_path: str | Path
) -> pd.DataFrame:
    targets = pd.read_csv(Path(target_path).expanduser().resolve())
    assignment = pd.read_csv(Path(split_path).expanduser().resolve())
    required = {
        "recommended_condition_id",
        "prompt_dataset",
        "request_rate",
        "target_power_kw",
    }
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(f"Condition target table missing columns: {sorted(missing)}")
    merged = targets.merge(
        assignment[["condition_id", "split"]],
        left_on="recommended_condition_id",
        right_on="condition_id",
        how="left",
        validate="one_to_one",
    )
    if merged["split"].isna().any():
        raise ValueError("Condition target table contains unassigned rows")
    return merged


def fit_m3_by_dataset(
    targets: pd.DataFrame, calibration_splits: Iterable[str]
) -> dict[str, M3Parameters]:
    allowed = set(calibration_splits)
    selected = targets.loc[targets["split"].isin(allowed)].copy()
    result: dict[str, M3Parameters] = {}
    for dataset, group in selected.groupby("prompt_dataset", observed=True):
        data = pd.DataFrame(
            {
                "request_rate": pd.to_numeric(group["request_rate"]),
                "power": pd.to_numeric(group["target_power_kw"]),
            }
        )
        model = fit_model("M3_logistic", data)
        result[str(dataset)] = M3Parameters(
            p0_kw=float(model.parameters["p0"]),
            delta_p_kw=float(model.parameters["delta_p"]),
            k=float(model.parameters["k"]),
            x0=float(model.parameters["x0"]),
        )
    return result


def _robust_sigma(values: np.ndarray) -> float:
    median = float(np.median(values))
    sigma = 1.4826 * float(np.median(np.abs(values - median)))
    if sigma <= np.finfo(float).eps:
        sigma = float(np.std(values, ddof=0))
    return max(sigma, 1e-9)


def _ar1_phi(values: np.ndarray, valid_pairs: np.ndarray | None = None) -> float:
    x0, x1 = values[:-1], values[1:]
    if valid_pairs is not None:
        x0, x1 = x0[valid_pairs], x1[valid_pairs]
    denominator = float(np.dot(x0, x0))
    if len(x0) < 4 or denominator <= np.finfo(float).eps:
        return 0.0
    return float(np.clip(np.dot(x0, x1) / denominator, -0.99, 0.995))


def _event_segments(
    values: np.ndarray,
    *,
    interval_s: float,
    threshold_mad: float,
    profile: PowerProfile,
) -> tuple[list[EventTemplate], dict[str, float | int]]:
    center = float(np.median(values))
    sigma = _robust_sigma(values)
    threshold = threshold_mad * sigma
    mask = np.abs(values - center) > threshold
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = changes[::2], changes[1::2]
    templates: list[EventTemplate] = []
    counts = {"burst": 0, "dip": 0}
    durations = {"burst": [], "dip": []}
    peaks = {"burst": [], "dip": []}
    for start, stop in zip(starts, stops):
        waveform = np.asarray(values[start:stop] - center, dtype=float)
        if waveform.size == 0:
            continue
        peak_index = int(np.argmax(np.abs(waveform)))
        sign = "burst" if waveform[peak_index] > 0 else "dip"
        duration = float(len(waveform) * interval_s)
        peak = float(waveform[peak_index])
        counts[sign] += 1
        durations[sign].append(duration)
        peaks[sign].append(peak)
        templates.append(
            EventTemplate(
                profile_id=profile.profile_id,
                prompt_dataset=profile.prompt_dataset,
                request_rate=profile.request_rate,
                threshold_mad=threshold_mad,
                sign=sign,
                waveform_kw=waveform,
                duration_s=duration,
                peak_kw=peak,
            )
        )
    duration_total = max(float((len(values) - 1) * interval_s), interval_s)
    return templates, {
        "burst_count": counts["burst"],
        "dip_count": counts["dip"],
        "burst_rate_per_s": counts["burst"] / duration_total,
        "dip_rate_per_s": counts["dip"] / duration_total,
        "burst_mean_duration_s": float(np.mean(durations["burst"])) if durations["burst"] else 0.0,
        "dip_mean_duration_s": float(np.mean(durations["dip"])) if durations["dip"] else 0.0,
        "burst_mean_peak_kw": float(np.mean(peaks["burst"])) if peaks["burst"] else 0.0,
        "dip_mean_peak_kw": float(np.mean(peaks["dip"])) if peaks["dip"] else 0.0,
        "event_fraction": float(np.mean(mask)),
    }


def build_residual_library(
    profiles: Sequence[PowerProfile],
    m3_parameters: Mapping[str, M3Parameters],
    calibration_splits: Iterable[str],
) -> ResidualLibrary:
    allowed = set(calibration_splits)
    rows: list[dict[str, Any]] = []
    residuals: dict[str, np.ndarray] = {}
    events: list[EventTemplate] = []
    selected_profiles: dict[str, PowerProfile] = {}
    for profile in profiles:
        if profile.split not in allowed:
            continue
        steady = float(m3_parameters[profile.prompt_dataset].predict(profile.request_rate))
        raw = profile.power_kw - steady
        centered = raw - np.mean(raw)
        residuals[profile.profile_id] = centered
        selected_profiles[profile.profile_id] = profile
        phi = _ar1_phi(centered)
        sigma = float(np.std(centered, ddof=0))
        for threshold_mad in EVENT_THRESHOLDS:
            templates, event_stats = _event_segments(
                centered,
                interval_s=profile.interval_s,
                threshold_mad=threshold_mad,
                profile=profile,
            )
            events.extend(templates)
            robust = _robust_sigma(centered)
            background_mask = np.abs(centered - np.median(centered)) <= threshold_mad * robust
            valid_pairs = background_mask[:-1] & background_mask[1:]
            background_phi = _ar1_phi(centered, valid_pairs)
            background_values = centered[background_mask]
            background_std = (
                float(np.std(background_values, ddof=0))
                if len(background_values) >= 4
                else sigma
            )
            rows.append(
                {
                    "profile_id": profile.profile_id,
                    "condition_id": profile.condition_id,
                    "prompt_dataset": profile.prompt_dataset,
                    "request_rate": profile.request_rate,
                    "split": profile.split,
                    "interval_s": profile.interval_s,
                    "m3_steady_power_kw": steady,
                    "profile_mean_power_kw": float(np.mean(profile.power_kw)),
                    "m3_bias_kw": float(np.mean(raw)),
                    "residual_std_kw": sigma,
                    "residual_ar1_phi": phi,
                    "residual_correlation_time_s": (
                        float(-profile.interval_s / math.log(phi))
                        if 0.0 < phi < 1.0
                        else float("nan")
                    ),
                    "event_threshold_mad": threshold_mad,
                    "background_std_kw": background_std,
                    "background_ar1_phi": background_phi,
                    **event_stats,
                }
            )
    statistics = pd.DataFrame(rows)
    if statistics.empty:
        raise ValueError(f"No calibration profiles for splits={sorted(allowed)}")
    return ResidualLibrary(
        statistics=statistics,
        residuals=residuals,
        events=events,
        profiles=selected_profiles,
    )


def _simulate_ar1(
    length: int, *, phi: float, stationary_std: float, rng: np.random.Generator
) -> np.ndarray:
    phi = float(np.clip(phi, -0.99, 0.995))
    stationary_std = max(float(stationary_std), 0.0)
    if stationary_std <= 0.0:
        return np.zeros(length)
    innovation_std = stationary_std * math.sqrt(max(1.0 - phi**2, 1e-8))
    values = np.empty(length, dtype=float)
    values[0] = rng.normal(0.0, stationary_std)
    noise = rng.normal(0.0, innovation_std, max(length - 1, 0))
    for index in range(1, length):
        values[index] = phi * values[index - 1] + noise[index - 1]
    return values


def _choose_template(
    library: ResidualLibrary,
    *,
    neighbour_ids: Sequence[str],
    threshold_mad: float,
    sign: str,
    rng: np.random.Generator,
) -> EventTemplate | None:
    allowed = set(neighbour_ids)
    candidates = [
        event
        for event in library.events
        if event.profile_id in allowed
        and event.threshold_mad == threshold_mad
        and event.sign == sign
    ]
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


def simulate_residual(
    model_name: str,
    *,
    length: int,
    interval_s: float,
    prompt_dataset: str,
    request_rate: float,
    library: ResidualLibrary,
    rng: np.random.Generator,
) -> np.ndarray:
    if model_name == "R0_m3_only":
        return np.zeros(length)

    neighbour_ids, neighbour_weights = library.neighbour_ids(
        prompt_dataset, request_rate
    )
    std = library.weighted_parameter(
        prompt_dataset, request_rate, "residual_std_kw"
    )
    if model_name == "R1_white_noise":
        result = rng.normal(0.0, max(std, 0.0), length)

    elif model_name == "R2_ar1":
        phi = library.weighted_parameter(
            prompt_dataset, request_rate, "residual_ar1_phi"
        )
        result = _simulate_ar1(length, phi=phi, stationary_std=std, rng=rng)

    elif model_name.startswith("R3_ar1_event_mad"):
        threshold_mad = float(model_name.rsplit("mad", 1)[1])
        background_std = library.weighted_parameter(
            prompt_dataset,
            request_rate,
            "background_std_kw",
            threshold_mad=threshold_mad,
        )
        background_phi = library.weighted_parameter(
            prompt_dataset,
            request_rate,
            "background_ar1_phi",
            threshold_mad=threshold_mad,
        )
        result = _simulate_ar1(
            length,
            phi=background_phi,
            stationary_std=background_std,
            rng=rng,
        )
        duration_s = max((length - 1) * interval_s, interval_s)
        for sign, rate_column in (
            ("burst", "burst_rate_per_s"),
            ("dip", "dip_rate_per_s"),
        ):
            rate = max(
                library.weighted_parameter(
                    prompt_dataset,
                    request_rate,
                    rate_column,
                    threshold_mad=threshold_mad,
                ),
                0.0,
            )
            count = int(rng.poisson(rate * duration_s))
            for _ in range(count):
                template = _choose_template(
                    library,
                    neighbour_ids=neighbour_ids,
                    threshold_mad=threshold_mad,
                    sign=sign,
                    rng=rng,
                )
                if template is None or len(template.waveform_kw) > length:
                    continue
                start = int(rng.integers(0, length - len(template.waveform_kw) + 1))
                result[start : start + len(template.waveform_kw)] += template.waveform_kw

    elif model_name == "R4_empirical_replay":
        donor_index = int(rng.choice(len(neighbour_ids), p=neighbour_weights))
        donor = np.asarray(library.residuals[neighbour_ids[donor_index]], dtype=float)
        if len(donor) < length:
            donor = np.tile(donor, int(np.ceil(length / len(donor))))
        shift = int(rng.integers(0, len(donor)))
        result = np.roll(donor, shift)[:length].copy()

    else:
        raise KeyError(model_name)

    return np.asarray(result, dtype=float) - float(np.mean(result))


def simulate_independent_residual_mean(
    model_name: str,
    *,
    n_units: int,
    length: int,
    interval_s: float,
    prompt_dataset: str,
    request_rate: float,
    library: ResidualLibrary,
    rng: np.random.Generator,
) -> np.ndarray:
    """Efficiently simulate the mean of independent equivalent residual units.

    Gaussian/AR(1) components scale exactly by ``1/sqrt(n_units)``.  For R3,
    independent Poisson event streams superpose to an event rate multiplied by
    ``n_units`` while each event amplitude is divided by ``n_units``.  This is
    distributionally equivalent to summing unit event streams and avoids a
    slow Python loop over hundreds or thousands of equivalent units.
    """
    if n_units < 1:
        raise ValueError("n_units must be at least one")
    if n_units == 1:
        return simulate_residual(
            model_name,
            length=length,
            interval_s=interval_s,
            prompt_dataset=prompt_dataset,
            request_rate=request_rate,
            library=library,
            rng=rng,
        )
    if model_name == "R0_m3_only":
        return np.zeros(length)

    std = library.weighted_parameter(
        prompt_dataset, request_rate, "residual_std_kw"
    )
    if model_name == "R1_white_noise":
        result = rng.normal(0.0, max(std, 0.0) / math.sqrt(n_units), length)

    elif model_name == "R2_ar1":
        phi = library.weighted_parameter(
            prompt_dataset, request_rate, "residual_ar1_phi"
        )
        result = _simulate_ar1(
            length,
            phi=phi,
            stationary_std=std / math.sqrt(n_units),
            rng=rng,
        )

    elif model_name.startswith("R3_ar1_event_mad"):
        threshold_mad = float(model_name.rsplit("mad", 1)[1])
        neighbour_ids, _ = library.neighbour_ids(prompt_dataset, request_rate)
        background_std = library.weighted_parameter(
            prompt_dataset,
            request_rate,
            "background_std_kw",
            threshold_mad=threshold_mad,
        )
        background_phi = library.weighted_parameter(
            prompt_dataset,
            request_rate,
            "background_ar1_phi",
            threshold_mad=threshold_mad,
        )
        result = _simulate_ar1(
            length,
            phi=background_phi,
            stationary_std=background_std / math.sqrt(n_units),
            rng=rng,
        )
        duration_s = max((length - 1) * interval_s, interval_s)
        for sign, rate_column in (
            ("burst", "burst_rate_per_s"),
            ("dip", "dip_rate_per_s"),
        ):
            unit_rate = max(
                library.weighted_parameter(
                    prompt_dataset,
                    request_rate,
                    rate_column,
                    threshold_mad=threshold_mad,
                ),
                0.0,
            )
            count = int(rng.poisson(unit_rate * duration_s * n_units))
            for _ in range(count):
                template = _choose_template(
                    library,
                    neighbour_ids=neighbour_ids,
                    threshold_mad=threshold_mad,
                    sign=sign,
                    rng=rng,
                )
                if template is None or len(template.waveform_kw) > length:
                    continue
                start = int(rng.integers(0, length - len(template.waveform_kw) + 1))
                result[start : start + len(template.waveform_kw)] += (
                    template.waveform_kw / n_units
                )

    else:
        # Empirical replay has no compact parametric superposition.  It is a
        # reference model, so retain an exact streaming average when requested.
        result = np.zeros(length)
        for _ in range(n_units):
            result += simulate_residual(
                model_name,
                length=length,
                interval_s=interval_s,
                prompt_dataset=prompt_dataset,
                request_rate=request_rate,
                library=library,
                rng=np.random.default_rng(int(rng.integers(0, 2**32 - 1))),
            )
        result /= n_units
    return np.asarray(result, dtype=float) - float(np.mean(result))


def temporal_features(values: np.ndarray, interval_s: float) -> dict[str, float]:
    residual = np.asarray(values, dtype=float)
    residual = residual[np.isfinite(residual)]
    if len(residual) < 20:
        raise ValueError("Residual requires at least 20 samples")
    residual = residual - np.mean(residual)
    std = float(np.std(residual, ddof=0))
    ramp = np.diff(residual) / interval_s
    acf = {
        f"acf_{str(lag).replace('.', 'p')}s": (
            _autocorrelation(residual, max(1, int(round(lag / interval_s))))
            if std > 1e-12
            else 0.0
        )
        for lag in ACF_LAGS_S
    }
    frequency, spectrum = welch(
        residual,
        fs=1.0 / interval_s,
        nperseg=min(1024, len(residual)),
        detrend="linear",
    )
    positive = frequency > 0
    total = float(np.sum(spectrum[positive]))
    low_frequency_share = (
        float(np.sum(spectrum[positive & (frequency <= 0.1)]) / total)
        if total > 0
        else 0.0
    )
    dummy = PowerProfile(
        profile_id="feature_only",
        condition_id="feature_only",
        prompt_dataset="feature_only",
        request_rate=0.0,
        split="feature_only",
        time_s=np.arange(len(residual)) * interval_s,
        power_kw=residual,
        interval_s=interval_s,
        measurement_origin="feature_only",
    )
    _, event = _event_segments(
        residual,
        interval_s=interval_s,
        threshold_mad=3.0,
        profile=dummy,
    )
    return {
        "std_kw": std,
        "p05_kw": float(np.quantile(residual, 0.05)),
        "p95_kw": float(np.quantile(residual, 0.95)),
        "p95_abs_ramp_kw_per_s": float(np.quantile(np.abs(ramp), 0.95)),
        "low_frequency_psd_share": low_frequency_share,
        **acf,
        **{key: float(value) for key, value in event.items()},
    }


def feature_errors(
    observed: np.ndarray,
    simulated: np.ndarray,
    interval_s: float,
) -> dict[str, float]:
    obs = temporal_features(observed, interval_s)
    sim = temporal_features(simulated, interval_s)
    eps = 1e-6
    obs_std = max(obs["std_kw"], eps)
    variability = abs(math.log((sim["std_kw"] + eps) / (obs["std_kw"] + eps)))
    quantile = (
        abs(sim["p05_kw"] - obs["p05_kw"])
        + abs(sim["p95_kw"] - obs["p95_kw"])
    ) / (2.0 * obs_std)
    wasserstein = float(wasserstein_distance(observed, simulated)) / obs_std
    distribution = 0.5 * quantile + 0.5 * wasserstein
    acf_columns = [f"acf_{str(lag).replace('.', 'p')}s" for lag in ACF_LAGS_S]
    temporal = float(
        np.sqrt(np.mean([(sim[column] - obs[column]) ** 2 for column in acf_columns]))
    )
    ramp = abs(
        math.log(
            (sim["p95_abs_ramp_kw_per_s"] + eps)
            / (obs["p95_abs_ramp_kw_per_s"] + eps)
        )
    )
    spectral = abs(sim["low_frequency_psd_share"] - obs["low_frequency_psd_share"])
    event_count = 0.5 * (
        abs(math.log((sim["burst_count"] + 1.0) / (obs["burst_count"] + 1.0)))
        + abs(math.log((sim["dip_count"] + 1.0) / (obs["dip_count"] + 1.0)))
    )
    event_duration = 0.5 * (
        abs(
            math.log(
                (sim["burst_mean_duration_s"] + interval_s)
                / (obs["burst_mean_duration_s"] + interval_s)
            )
        )
        + abs(
            math.log(
                (sim["dip_mean_duration_s"] + interval_s)
                / (obs["dip_mean_duration_s"] + interval_s)
            )
        )
    )
    event = 0.6 * event_count + 0.4 * event_duration
    score = (
        0.20 * variability
        + 0.15 * distribution
        + 0.25 * temporal
        + 0.20 * ramp
        + 0.10 * spectral
        + 0.10 * event
    )
    return {
        "overall_score": score,
        "variability_error": variability,
        "distribution_error": distribution,
        "acf_error": temporal,
        "ramp_error": ramp,
        "spectral_error": spectral,
        "event_error": event,
    }


def _stable_seed(base_seed: int, *parts: str) -> int:
    text = "|".join(parts).encode("utf-8")
    return int((base_seed + zlib.crc32(text)) % (2**32 - 1))


def evaluate_models(
    *,
    profiles: Sequence[PowerProfile],
    evaluation_split: str,
    m3_parameters: Mapping[str, M3Parameters],
    library: ResidualLibrary,
    draws: int,
    seed: int,
    model_names: Sequence[str] = MODEL_NAMES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []
    selected_profiles = [profile for profile in profiles if profile.split == evaluation_split]
    for profile in tqdm(selected_profiles, desc=f"Evaluating {evaluation_split}"):
        steady = float(m3_parameters[profile.prompt_dataset].predict(profile.request_rate))
        observed = profile.power_kw - steady
        observed_centered = observed - np.mean(observed)
        for model_name in model_names:
            repetitions = 1 if model_name == "R0_m3_only" else draws
            model_errors: list[dict[str, float]] = []
            first_simulation: np.ndarray | None = None
            for draw in range(repetitions):
                rng = np.random.default_rng(
                    _stable_seed(seed, evaluation_split, profile.profile_id, model_name, str(draw))
                )
                simulated = simulate_residual(
                    model_name,
                    length=len(observed_centered),
                    interval_s=profile.interval_s,
                    prompt_dataset=profile.prompt_dataset,
                    request_rate=profile.request_rate,
                    library=library,
                    rng=rng,
                )
                if first_simulation is None:
                    first_simulation = simulated
                model_errors.append(
                    feature_errors(observed_centered, simulated, profile.interval_s)
                )
            errors = pd.DataFrame(model_errors).mean(numeric_only=True).to_dict()
            rows.append(
                {
                    "evaluation_split": evaluation_split,
                    "profile_id": profile.profile_id,
                    "prompt_dataset": profile.prompt_dataset,
                    "request_rate": profile.request_rate,
                    "model": model_name,
                    "draws": repetitions,
                    "m3_steady_power_kw": steady,
                    "observed_mean_power_kw": float(np.mean(profile.power_kw)),
                    "m3_mean_bias_kw": float(np.mean(observed)),
                    **errors,
                }
            )
            if profile.request_rate in {50.0, 90.0, 100.0} and model_name in {
                "R0_m3_only",
                "R2_ar1",
                "R3_ar1_event_mad3",
                "R3_ar1_event_mad4",
                "R3_ar1_event_mad5",
                "R4_empirical_replay",
            }:
                assert first_simulation is not None
                for index, time_s in enumerate(profile.time_s):
                    representative_rows.append(
                        {
                            "evaluation_split": evaluation_split,
                            "profile_id": profile.profile_id,
                            "prompt_dataset": profile.prompt_dataset,
                            "request_rate": profile.request_rate,
                            "model": model_name,
                            "time_s": time_s,
                            "m3_steady_power_kw": steady,
                            "measured_power_kw": profile.power_kw[index],
                            "simulated_power_kw": steady + first_simulation[index],
                        }
                    )
    per_profile = pd.DataFrame(rows)
    representative = pd.DataFrame(representative_rows)
    return per_profile, representative


def aggregate_metrics(per_profile: pd.DataFrame) -> pd.DataFrame:
    components = [
        "overall_score",
        "variability_error",
        "distribution_error",
        "acf_error",
        "ramp_error",
        "spectral_error",
        "event_error",
    ]
    rows = []
    for model, group in per_profile.groupby("model", observed=True):
        row: dict[str, Any] = {
            "model": model,
            "profile_count": int(group["profile_id"].nunique()),
            "mean_abs_m3_bias_kw": float(group["m3_mean_bias_kw"].abs().mean()),
        }
        for column in components:
            row[f"mean_{column}"] = float(group[column].mean())
            row[f"median_{column}"] = float(group[column].median())
            row[f"sem_{column}"] = float(
                group[column].std(ddof=1) / math.sqrt(len(group))
            )
        row["ci95_low_overall_score"] = max(
            0.0,
            row["mean_overall_score"] - 1.96 * row["sem_overall_score"],
        )
        row["ci95_high_overall_score"] = (
            row["mean_overall_score"] + 1.96 * row["sem_overall_score"]
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_overall_score", kind="stable")


def choose_residual_model(validation_metrics: pd.DataFrame) -> tuple[str, str]:
    eligible = validation_metrics.loc[
        validation_metrics["model"].isin(DEPLOYABLE_MODELS)
    ].copy()
    winner = eligible.sort_values(
        ["mean_overall_score", "mean_acf_error", "mean_ramp_error"], kind="stable"
    ).iloc[0]
    return (
        str(winner["model"]),
        "Empirical replay를 제외한 deployable 후보 중 validation 평균 통계재현 score 최소",
    )


def weight_sensitivity(
    metrics: pd.DataFrame,
    *,
    samples: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Test whether model ranking depends on one chosen component weighting.

    We sample weights uniformly from the six-dimensional simplex.  R4 remains
    excluded because it reuses measured donor traces and is a reference rather
    than a deployable parametric generator.
    """
    component_columns = [
        "mean_variability_error",
        "mean_distribution_error",
        "mean_acf_error",
        "mean_ramp_error",
        "mean_spectral_error",
        "mean_event_error",
    ]
    eligible = metrics.loc[metrics["model"].isin(DEPLOYABLE_MODELS)].copy()
    values = eligible[component_columns].to_numpy(float)
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(component_columns)), size=samples)
    winner = np.argmin(weights @ values.T, axis=1)
    counts = pd.Series(eligible["model"].to_numpy()[winner]).value_counts()
    return pd.DataFrame(
        {
            "model": eligible["model"],
            "weight_sample_count": [int(counts.get(name, 0)) for name in eligible["model"]],
            "selection_fraction": [
                float(counts.get(name, 0) / samples) for name in eligible["model"]
            ],
            "weight_sampling_design": "Dirichlet(1,...,1) over six score components",
        }
    ).sort_values("selection_fraction", ascending=False, kind="stable")


def paired_model_comparison(
    per_profile: pd.DataFrame,
    selected_model: str,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    selected = per_profile.loc[
        per_profile["model"].eq(selected_model),
        ["profile_id", "overall_score"],
    ].rename(columns={"overall_score": "selected_score"})
    rows = []
    rng = np.random.default_rng(seed)
    for model, group in per_profile.loc[
        per_profile["model"].ne(selected_model)
    ].groupby("model", observed=True):
        paired = selected.merge(
            group[["profile_id", "overall_score"]].rename(
                columns={"overall_score": "competitor_score"}
            ),
            on="profile_id",
            validate="one_to_one",
        )
        difference = paired["competitor_score"].to_numpy(float) - paired[
            "selected_score"
        ].to_numpy(float)
        indices = rng.integers(0, len(difference), size=(bootstrap_samples, len(difference)))
        boot_mean = difference[indices].mean(axis=1)
        rows.append(
            {
                "selected_model": selected_model,
                "competitor_model": model,
                "profile_count": len(difference),
                "mean_score_advantage": float(np.mean(difference)),
                "median_score_advantage": float(np.median(difference)),
                "selected_win_fraction": float(np.mean(difference > 0.0)),
                "bootstrap_ci95_low": float(np.quantile(boot_mean, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(boot_mean, 0.975)),
                "interpretation": "positive advantage favors selected model",
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mean_score_advantage", ascending=False, kind="stable"
    )


def _model_parameter_rows(
    stage: str, m3_parameters: Mapping[str, M3Parameters]
) -> list[dict[str, Any]]:
    rows = []
    for dataset, parameter in m3_parameters.items():
        for name, value in {
            "p0": parameter.p0_kw,
            "delta_p": parameter.delta_p_kw,
            "k": parameter.k,
            "x0": parameter.x0,
        }.items():
            rows.append(
                {
                    "stage": stage,
                    "prompt_dataset": dataset,
                    "parameter": name,
                    "value": value,
                }
            )
    return rows


def _plot_score_bars(metrics: pd.DataFrame, title: str, output: Path) -> None:
    ordered = metrics.sort_values("mean_overall_score", ascending=True)
    colors = ["#9ca3af" if name == "R4_empirical_replay" else "#2563eb" for name in ordered["model"]]
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    ax.barh(ordered["model"], ordered["mean_overall_score"], color=colors)
    ax.set_xlabel("Mean statistical reconstruction score (lower is better)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_component_heatmap(metrics: pd.DataFrame, title: str, output: Path) -> None:
    columns = [
        "mean_variability_error",
        "mean_distribution_error",
        "mean_acf_error",
        "mean_ramp_error",
        "mean_spectral_error",
        "mean_event_error",
    ]
    ordered = metrics.sort_values("mean_overall_score").set_index("model")
    values = ordered[columns].to_numpy(float)
    normalized = values / np.maximum(np.nanmax(values, axis=0, keepdims=True), 1e-12)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    image = ax.imshow(normalized, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_yticks(range(len(ordered.index)), ordered.index)
    labels = ["variability", "distribution", "ACF", "ramp", "PSD", "event"]
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_title(title + " (column-normalized)")
    fig.colorbar(image, ax=ax, label="Relative error")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_representative_test(
    traces: pd.DataFrame, selected_model: str, output: Path
) -> None:
    working = traces.loc[
        traces["evaluation_split"].eq("test")
        & traces["request_rate"].eq(90.0)
        & traces["model"].isin({selected_model, "R4_empirical_replay"})
    ].copy()
    datasets = list(working["prompt_dataset"].drop_duplicates())
    if not datasets:
        return
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 4.2 * len(datasets)), dpi=150, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets, strict=True):
        part = working.loc[working["prompt_dataset"].eq(dataset)]
        observed = part.loc[part["model"].eq(selected_model)]
        ax.plot(observed["time_s"], observed["measured_power_kw"], color="#111827", lw=1.0, label="measured")
        for model, color in ((selected_model, "#2563eb"), ("R4_empirical_replay", "#f97316")):
            model_part = part.loc[part["model"].eq(model)]
            if not model_part.empty:
                ax.plot(model_part["time_s"], model_part["simulated_power_kw"], color=color, lw=0.9, alpha=0.9, label=model)
        ax.axhline(observed["m3_steady_power_kw"].iloc[0], color="#059669", lw=1.8, label="M3 steady")
        ax.set_title(f"{dataset} | 90 request/s high-rate test")
        ax.set_ylabel("IT power (kW)")
        ax.grid(alpha=0.22)
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


def _report(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    selected_model: str,
    selection_basis: str,
    draws: int,
    robustness: pd.DataFrame,
) -> str:
    view_columns = [
        "model",
        "mean_overall_score",
        "mean_variability_error",
        "mean_acf_error",
        "mean_ramp_error",
        "mean_event_error",
    ]
    validation_table = _markdown_table(validation[view_columns].round(4).reset_index(drop=True))
    test_table = _markdown_table(test[view_columns].round(4).reset_index(drop=True))
    robustness_table = _markdown_table(
        robustness[["model", "selection_fraction"]].round(4).reset_index(drop=True)
    )
    threshold = selected_model.rsplit("mad", 1)[1] if "event_mad" in selected_model else "해당 없음"
    return f"""# NLR M3 residual 동특성 모델 비교

## 결론

- Validation에서 선정된 deployable 모델: **{selected_model}**
- Event threshold: **{threshold} MAD**
- 선정 기준: {selection_basis}
- Stochastic Monte Carlo draw: profile/model당 {draws}회
- R4 empirical replay는 실제 residual donor를 이용하는 비모수 reference이므로 최종 모델 선정 대상에서 제외함.

## Validation 결과

{validation_table}

## High-rate extrapolation test 결과

{test_table}

## Score 가중치 민감도

{robustness_table}

- 여섯 score component의 가중치를 simplex에서 100,000회 무작위 추출함.
- 특정 가중치 한 조합에만 의존한 모델 선정인지 확인하기 위한 탐색적 robustness 분석임.

## Score 정의

- 평균·시점별 RMSE가 아니라 residual의 변동성, 분포, ACF, ramp, PSD 및 event 재현오차를 평가함.
- 가중치: variability 0.20, distribution 0.15, ACF 0.25, ramp 0.20, PSD 0.10, event 0.10.
- Score가 낮을수록 실제 NLR residual의 통계적·시간적 특성을 잘 재현함.

## 데이터 누수 방지

1. Validation: train request-rate만으로 M3와 residual parameter를 보정함.
2. Test: train+validation으로 재보정한 뒤 81–100 request/s에 적용함.
3. 평가 profile 자체는 parameter estimation이나 empirical replay donor로 사용하지 않음.

## 해석상 주의

- AR(1)/OU parameter는 0.1초 보간 profile의 출력 상관성을 나타내며 물리적 입력-출력 시정수가 아님.
- NLR online-rate 조건은 condition당 반복실험이 없으므로 run-to-run 불확실성을 식별한 결과가 아님.
- Event는 centered residual의 MAD threshold를 넘는 후보 구간이며 원인이 확인된 물리 event가 아님.
- 현재 결과는 서버/job 수준 IT power이다. MW급 AIDC 적용 전 비동기 서버 aggregation과 scaling 검증이 필요함.
"""


def run_residual_model_comparison(
    *,
    dataset_root: str | Path,
    manifest_path: str | Path,
    condition_target_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    draws: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    if draws < 2:
        raise ValueError("draws must be at least 2 for stochastic model comparison")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)

    profiles = load_profiles(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split_path=split_path,
    )
    targets = load_condition_targets(condition_target_path, split_path)

    validation_m3 = fit_m3_by_dataset(targets, {"train"})
    validation_library = build_residual_library(profiles, validation_m3, {"train"})
    validation_profiles, validation_traces = evaluate_models(
        profiles=profiles,
        evaluation_split="validation",
        m3_parameters=validation_m3,
        library=validation_library,
        draws=draws,
        seed=seed,
    )
    validation_metrics = aggregate_metrics(validation_profiles)
    selected_model, selection_basis = choose_residual_model(validation_metrics)
    validation_robustness = weight_sensitivity(
        validation_metrics, samples=100_000, seed=seed
    )
    validation_paired = paired_model_comparison(
        validation_profiles, selected_model, seed=seed
    )

    test_m3 = fit_m3_by_dataset(targets, {"train", "validation"})
    test_library = build_residual_library(
        profiles, test_m3, {"train", "validation"}
    )
    test_profiles, test_traces = evaluate_models(
        profiles=profiles,
        evaluation_split="test",
        m3_parameters=test_m3,
        library=test_library,
        draws=draws,
        seed=seed,
    )
    test_metrics = aggregate_metrics(test_profiles)
    test_paired = paired_model_comparison(
        test_profiles, selected_model, seed=seed + 1
    )
    traces = pd.concat([validation_traces, test_traces], ignore_index=True)

    tables = {
        "validation_per_profile_metrics": validation_profiles,
        "validation_model_metrics": validation_metrics,
        "test_per_profile_metrics": test_profiles,
        "test_model_metrics": test_metrics,
        "calibration_profile_parameters": test_library.statistics,
        "representative_simulations": traces,
        "m3_parameters_by_stage": pd.DataFrame(
            _model_parameter_rows("validation_train_only", validation_m3)
            + _model_parameter_rows("test_train_plus_validation", test_m3)
        ),
        "validation_weight_sensitivity": validation_robustness,
        "validation_paired_comparison": validation_paired,
        "test_paired_comparison": test_paired,
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
        if not table.empty:
            table.to_parquet(output / f"{name}.parquet", index=False)

    event_summary = pd.DataFrame(
        [
            {
                "profile_id": event.profile_id,
                "prompt_dataset": event.prompt_dataset,
                "request_rate": event.request_rate,
                "threshold_mad": event.threshold_mad,
                "sign": event.sign,
                "duration_s": event.duration_s,
                "peak_kw": event.peak_kw,
            }
            for event in test_library.events
        ]
    )
    event_summary.to_csv(output / "event_template_catalog.csv", index=False, encoding="utf-8-sig")
    if not event_summary.empty:
        event_summary.to_parquet(output / "event_template_catalog.parquet", index=False)

    selected_test = test_metrics.loc[test_metrics["model"].eq(selected_model)].iloc[0]
    decision = {
        "selected_residual_model": selected_model,
        "selection_basis": selection_basis,
        "validation_mean_score": float(
            validation_metrics.loc[
                validation_metrics["model"].eq(selected_model), "mean_overall_score"
            ].iloc[0]
        ),
        "high_rate_test_mean_score": float(selected_test["mean_overall_score"]),
        "empirical_replay_is_reference_only": True,
        "causal_plant_time_constant_claimed": False,
        "draws_per_profile_model": draws,
        "validation_selection_fraction_under_random_weights": float(
            validation_robustness.loc[
                validation_robustness["model"].eq(selected_model),
                "selection_fraction",
            ].iloc[0]
        ),
        "seed": seed,
    }
    (output / "selected_residual_model.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "residual_model_comparison_report_ko.md").write_text(
        _report(
            validation_metrics,
            test_metrics,
            selected_model,
            selection_basis,
            draws,
            validation_robustness,
        ),
        encoding="utf-8",
    )

    _plot_score_bars(
        validation_metrics,
        "Residual model comparison: interpolation validation",
        figures / "validation_model_scores.png",
    )
    _plot_score_bars(
        test_metrics,
        "Residual model comparison: high-rate extrapolation test",
        figures / "test_model_scores.png",
    )
    _plot_component_heatmap(
        test_metrics,
        "High-rate test feature errors",
        figures / "test_feature_error_heatmap.png",
    )
    _plot_representative_test(
        traces, selected_model, figures / "representative_high_rate_test.png"
    )
    return {
        "tables": tables,
        "decision": decision,
        "output_dir": output,
    }
