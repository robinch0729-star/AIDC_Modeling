"""NLR scheduler-state diagnostics; no serving capacity or transition model is fitted.

Observed scheduler stocks and token rates are kept distinct from inferred request
departure rates. A configured arrival rate is NOT an observed admitted rate.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STAMP = re.compile(r"\b(\d{2}-\d{2} \d{2}:\d{2}:\d{2})\b")
METRIC = re.compile(
    r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs")
WINDOWS = {"primary_60_240": (60, 240), "early_60_150": (60, 150),
           "late_150_240": (150, 240), "alternative_90_270": (90, 270)}
SERVICES = ("InstructCoder", "mlperf-inference-llama2-data")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def service_name(dataset_path: str) -> str:
    name = dataset_path.rsplit("/", 1)[-1]
    if name not in SERVICES:
        raise ValueError(f"Unrecognized dataset: {dataset_path}")
    return name


def parse_metric_line(text: str, year: int, source_line: int) -> dict | None:
    clean = ANSI.sub("", text)
    match, stamp = METRIC.search(clean), STAMP.search(clean)
    if match is None or stamp is None:
        return None
    row = dict(timestamp=f"{year}-{stamp.group(1)}", prompt_tokens_per_s=float(match[1]),
               generation_tokens_per_s=float(match[2]), running_requests=int(match[3]),
               waiting_requests=int(match[4]), source_line=source_line)
    for label, key in (("GPU KV cache usage", "gpu_kv_cache_pct"),
                       ("Prefix cache hit rate", "prefix_cache_hit_pct")):
        item = re.search(re.escape(label) + r": ([\d.]+)%", clean)
        row[key] = float(item[1]) if item else np.nan
    return row


def scan_server(path: Path, year: int) -> tuple[pd.DataFrame, dict, list[dict]]:
    """Binary line counting matches the previous raw-file audit exactly."""
    h, rows, examples = hashlib.sha256(), [], []
    markers = {"metric_markers": 0, "unparsed_metric_markers": 0,
               "error_or_exception_markers": 0, "abort_or_preempt_markers": 0}
    for line_n, raw in enumerate(path.open("rb"), 1):
        h.update(raw)
        # Avoid decoding millions of HTTP access lines that contain no diagnostics.
        is_metric = b"Avg prompt throughput:" in raw
        if is_metric:
            markers["metric_markers"] += 1
            text = raw.decode("utf-8", errors="replace")
            row = parse_metric_line(text, year, line_n)
            if row is None:
                markers["unparsed_metric_markers"] += 1
            else:
                rows.append(row)
        for key, pattern in (("error_or_exception_markers", rb"\bERROR\b|Exception"),
                             ("abort_or_preempt_markers", rb"[Aa]borted|[Pp]reempt")):
            if re.search(pattern, raw):
                markers[key] += 1
                if markers[key] <= 3:
                    examples.append(dict(category=key, source_line=line_n,
                                         excerpt=ANSI.sub("", raw.decode("utf-8", "replace"))[:1500]))
        if any(p in raw for p in (b"vLLM API server version", b"Initializing a V1 LLM engine",
                                  b"Shutting down")):
            examples.append(dict(category="runtime_configuration", source_line=line_n,
                                 excerpt=ANSI.sub("", raw.decode("utf-8", "replace"))[:3000]))
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame.timestamp)
    return frame, dict(sha256=h.hexdigest(), bytes=path.stat().st_size, lines=line_n,
                       parsed_metrics=len(rows), **markers), examples


def linear_slope(t, y) -> float:
    t, y = np.asarray(t, float), np.asarray(y, float)
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("At least two strictly increasing timestamps are required")
    tc = t - t.mean()
    return float(np.dot(tc, y-y.mean()) / np.dot(tc, tc))


def time_average(t, y) -> float:
    """Trapezoid mean only over the observed first-to-last snapshot span.

    This is an integration convention between sparse snapshots, not a claim that
    instantaneous queue states vary linearly or that unobserved endpoints are known.
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("At least two strictly increasing timestamps are required")
    return float(np.sum(.5 * (y[1:] + y[:-1]) * np.diff(t)) / (t[-1] - t[0]))


def residence_screen(stats: dict, drift_tolerance: float = .02) -> bool:
    """Exploratory low-drift screen, not a stability/SLO statistical test."""
    return bool(stats["metric_count"] >= 6 and stats["max_metric_gap_s"] <= 20
                and stats["waiting_max"] == 0
                and abs(stats["unfinished_slope_rps"]) <= drift_tolerance*stats["configured_rps"]
                and abs(stats["unfinished_endpoint_slope_rps"]) <= drift_tolerance*stats["configured_rps"])


def diagnose_window(frame: pd.DataFrame, configured_rps: float) -> dict:
    if configured_rps <= 0 or len(frame) < 2:
        raise ValueError("Positive offered rate and >=2 snapshots required")
    t = (frame.timestamp - frame.timestamp.iloc[0]).dt.total_seconds().to_numpy()
    running = frame.running_requests.to_numpy(float)
    waiting = frame.waiting_requests.to_numpy(float)
    unfinished = running + waiting
    span = t[-1] - t[0]
    slope = linear_slope(t, unfinished)
    row = dict(configured_rps=float(configured_rps), metric_count=len(frame),
               observed_span_s=float(span), first_snapshot=str(frame.timestamp.iloc[0]),
               last_snapshot=str(frame.timestamp.iloc[-1]), max_metric_gap_s=float(np.diff(t).max()),
               first_source_line=int(frame.source_line.iloc[0]), last_source_line=int(frame.source_line.iloc[-1]),
               running_mean=time_average(t, running), running_max=float(running.max()),
               waiting_mean=time_average(t, waiting), waiting_max=float(waiting.max()),
               waiting_positive_snapshot_fraction=float(np.mean(waiting > 0)),
               waiting_positive_all_snapshots=bool(np.all(waiting > 0)),
               unfinished_first=float(unfinished[0]), unfinished_last=float(unfinished[-1]),
               unfinished_mean=time_average(t, unfinished), unfinished_slope_rps=slope,
               running_slope_rps=linear_slope(t, running), waiting_slope_rps=linear_slope(t, waiting),
               unfinished_endpoint_slope_rps=float((unfinished[-1]-unfinished[0])/span),
               max_observed_stock_drop=float(max(0, -np.diff(unfinished).min())),
               conditional_departure_rps=float(configured_rps-slope),
               conditional_endpoint_departure_rps=float(configured_rps-(unfinished[-1]-unfinished[0])/span),
               # Token rates are interval averages at each logging timestamp, not stocks.
               prompt_tps_snapshot_mean=float(frame.prompt_tokens_per_s.mean()),
               generation_tps_snapshot_mean=float(frame.generation_tokens_per_s.mean()),
               running_over_configured_rate_s=time_average(t, running)/configured_rps)
    row["low_drift_screen_2pct"] = residence_screen(row)
    row["conditional_residence_proxy_s"] = row["running_over_configured_rate_s"] if row["low_drift_screen_2pct"] else np.nan
    return row


def diagnose_runs(runs: pd.DataFrame, metrics: pd.DataFrame, power: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows, boundaries = [], []
    for _, run in runs.iterrows():
        start, end = pd.Timestamp(run.run_start), pd.Timestamp(run.run_end)
        service = service_name(run["dataset-path"])
        common = dict(run_id=str(run.id), service=service, configured_rps=float(run.request_rate))
        full = metrics[(metrics.timestamp >= start) & (metrics.timestamp < end)].copy()
        rel = (full.timestamp-start).dt.total_seconds()
        stock = full.running_requests + full.waiting_requests
        post_peak = full.loc[stock.idxmax():] if len(full) else full
        after_zero = post_peak[(post_peak.running_requests+post_peak.waiting_requests).eq(0)]
        boundaries.append(dict(**common, full_run_metric_count=len(full), full_run_waiting_max=int(full.waiting_requests.max()),
            full_run_unfinished_max=int(stock.max()), first_running=int(full.running_requests.iloc[0]),
            first_waiting=int(full.waiting_requests.iloc[0]), last_running=int(full.running_requests.iloc[-1]),
            last_waiting=int(full.waiting_requests.iloc[-1]),
            first_zero_after_max_s=float((after_zero.timestamp.iloc[0]-start).total_seconds()) if len(after_zero) else np.nan,
            max_stock_drop_between_snapshots=int(max(0, -stock.diff().min())),
            peak_source_line=int(full.loc[stock.idxmax(), "source_line"])))
        for name, (left, right) in WINDOWS.items():
            part = full[(rel >= left) & (rel < right)]
            row = diagnose_window(part, run.request_rate)
            row.update(common, window=name, window_start_s=left, window_end_s=right,
                       requested_window_s=right-left)
            p = power[(power.timestamp >= start+pd.Timedelta(seconds=left)) &
                      (power.timestamp < start+pd.Timedelta(seconds=right))]
            row.update(power_seconds=len(p), gpu_mean_kw=float(p.gpu_w.mean()/1000),
                       cpu_reported_domains_mean_kw=float(p.cpu_reported_domains_w.mean()/1000))
            windows.append(row)
    result = pd.DataFrame(windows)
    result["low_drift_all_windows_2pct"] = result.groupby("run_id")["low_drift_screen_2pct"].transform("all")
    result.loc[~result.low_drift_all_windows_2pct, "conditional_residence_proxy_s"] = np.nan
    return result, pd.DataFrame(boundaries)


def align_logged_intervals(runs: pd.DataFrame, metrics: pd.DataFrame, power: pd.DataFrame) -> pd.DataFrame:
    """Pair token logger intervals with wall-clock power; never interpolate queue states."""
    result = []
    for _, run in runs.iterrows():
        start, end = pd.Timestamp(run.run_start), pd.Timestamp(run.run_end)
        samples = metrics[(metrics.timestamp >= start) & (metrics.timestamp < end)]
        previous = None
        for _, sample in samples.iterrows():
            if previous is not None:
                dt = (sample.timestamp-previous).total_seconds()
                p = power[(power.timestamp >= previous) & (power.timestamp < sample.timestamp)]
                result.append(dict(run_id=run.id, service=service_name(run["dataset-path"]),
                    configured_rps=run.request_rate, interval_start=str(previous), interval_end=str(sample.timestamp),
                    relative_end_s=(sample.timestamp-start).total_seconds(), interval_s=dt,
                    prompt_tokens_per_s=sample.prompt_tokens_per_s, generation_tokens_per_s=sample.generation_tokens_per_s,
                    running_at_end=sample.running_requests, waiting_at_end=sample.waiting_requests,
                    source_line=sample.source_line, gpu_mean_kw=p.gpu_w.mean()/1000,
                    cpu_reported_domains_mean_kw=p.cpu_reported_domains_w.mean()/1000,
                    power_seconds=len(p)))
            previous = sample.timestamp
    return pd.DataFrame(result)


def finite_diagnostics(meta: pd.DataFrame, results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if meta.id.duplicated().any() or results.id.duplicated().any():
        raise ValueError("Duplicate finite benchmark IDs")
    merged = results.merge(meta[["id", "dataset-path", "hf-output-len", "request_rate"]], on="id", suffixes=("", "_meta"), validate="one_to_one")
    if len(merged) != len(results) or not np.allclose(merged.request_rate, merged.request_rate_meta):
        raise ValueError("Finite result/metadata mismatch")
    merged["service"] = merged["dataset-path"].map(service_name)
    merged["mean_e2el_s"] = merged.mean_e2el_ms/1000
    merged["observed_output_tokens_per_completed_request"] = merged.total_output_tokens/merged.completed
    merged["request_throughput_recalculated"] = merged.completed/merged.duration
    merged["throughput_identity_error"] = merged.request_throughput-merged.request_throughput_recalculated
    cols = ["id", "service", "hf-output-len", "request_rate", "completed", "duration", "request_throughput",
            "mean_e2el_s", "observed_output_tokens_per_completed_request", "throughput_identity_error"]
    grouped = merged.groupby(["service", "hf-output-len", "request_rate"], as_index=False).agg(
        runs=("id", "size"), mean_e2el_s=("mean_e2el_s", "mean"), min_e2el_s=("mean_e2el_s", "min"),
        max_e2el_s=("mean_e2el_s", "max"), mean_completed_rps=("request_throughput", "mean"),
        mean_output_tokens_per_request=("observed_output_tokens_per_completed_request", "mean"))
    return merged[cols], grouped, dict(metadata_runs=len(meta), result_runs=len(results),
        missing_result_ids=sorted(set(meta.id)-set(results.id)),
        max_throughput_identity_error=float(abs(merged.throughput_identity_error).max()),
        completed_min=int(merged.completed.min()), completed_max=int(merged.completed.max()),
        mean_e2el_s_min=float(merged.mean_e2el_s.min()), mean_e2el_s_max=float(merged.mean_e2el_s.max()))
