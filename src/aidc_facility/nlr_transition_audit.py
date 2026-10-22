"""Read-only audit of NLR raw logs for identifying load transition dynamics.

No transition constant is fitted here.  A timestamped benchmark boundary is
not a timestamped request-rate step; API response access lines are not arrival
events.  Every supplied raw file is scanned to EOF with a SHA-256 digest.
Power streams are reduced to one-second diagnostic means during streaming,
not retained as enormous in-memory raw matrices.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import ast
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STAMP = re.compile(r"(?:\d{4}[-_]\d{2}[-_]\d{2}[T_ ]|\b\d{2}-\d{2} )\d{2}:\d{2}:\d{2}(?:\.\d+)?")
SHORT_STAMP = re.compile(r"\b(\d{2}-\d{2} \d{2}:\d{2}:\d{2})\b")
METRICS = re.compile(
    r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs"
)
REQUEST_DETAIL_KEYS = {
    "arrival_times", "request_timestamps", "request_start_times", "start_times",
    "request_end_times", "completion_times", "send_times", "request_arrivals",
}


def parse_stamp(value: str) -> datetime:
    """Preserve naive source wall-clock; never silently assume UTC or local TZ."""
    return datetime.fromisoformat(value.replace("_", "T"))


def namespace_fields(line: str) -> dict:
    """Extract scalar fields, without evaluating the Python Namespace text."""
    result = {}
    keys = (
        "request_rate", "num_prompts", "dataset_path", "burstiness", "save_detailed",
        "save_result", "ramp_up_strategy", "ramp_up_start_rps", "ramp_up_end_rps",
        "hf_output_len", "model", "seed",
    )
    for key in keys:
        match = re.search(r"\b" + key + r"=(('[^']*')|(\"[^\"]*\")|[^,)]+)", line)
        if match:
            token = match.group(1).strip()
            try:
                result[key] = ast.literal_eval(token)
            except (ValueError, SyntaxError):
                result[key] = token
    identity = re.search(r"metadata=\['id=([^']+)'\]", line)
    result["id"] = identity.group(1) if identity else None
    return result


def _delta_summary(histogram: Counter) -> dict:
    if not histogram:
        return {}
    total = sum(histogram.values())
    c = 0
    median = None
    for dt, n in sorted(histogram.items()):
        c += n
        if median is None and c >= (total + 1) / 2:
            median = dt / 1e6
    return {"count": total, "median_s_microsecond_rounded": median,
            "min_s": min(histogram) / 1e6, "max_s": max(histogram) / 1e6,
            "nonpositive_count": sum(n for dt, n in histogram.items() if dt <= 0),
            "above_0p2s_count": sum(n for dt, n in histogram.items() if dt > 200000)}


class Evidence:
    """Bounded examples per file/category, with full-file occurrence counts."""
    def __init__(self):
        self.counts = Counter()
        self.examples = defaultdict(list)

    def add(self, path: str, line: int, category: str, excerpt: str, interpretation: str = ""):
        key = (path, category)
        self.counts[key] += 1
        record = {"relative_path": path, "line": line, "category": category,
                  "excerpt": excerpt.strip()[:3500], "interpretation": interpretation}
        bucket = self.examples[key]
        if len(bucket) < 3:
            bucket.append(record)
        else:
            bucket[2] = record  # first two and last occurrence, not a biased-only prefix.

    def rows(self):
        rows = []
        for key, examples in sorted(self.examples.items()):
            for row in examples:
                rows.append(dict(row, full_file_category_occurrences=self.counts[key]))
        return rows


def scan_power(path: Path, raw_root: Path, evidence: Evidence):
    """Scan all lines; average raw CPU/GPU streams separately into wallclock seconds."""
    h = hashlib.sha256()
    size = 0
    count = malformed = comments = 0
    first = last = previous = None
    columns = []
    selected = []
    factor = 1.0
    histogram = Counter()
    sums = defaultdict(float)
    samples = Counter()
    rel = path.relative_to(raw_root).as_posix()
    minimum = math.inf
    maximum = -math.inf
    line_n = 0
    with path.open("rb") as file:
        for line_n, raw in enumerate(file, 1):
            h.update(raw)
            size += len(raw)
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            if text.startswith("#"):
                comments += 1
                if "timestamp" in text and "reading-time" in text:
                    columns = text.lstrip("# ").split()
                    selected = [i for i, name in enumerate(columns) if name.endswith("[mW]") or name.endswith("[W]")]
                    factor = .001 if any(columns[i].endswith("[mW]") for i in selected) else 1.0
                    evidence.add(rel, line_n, "power_header", text,
                                 "CPU package/core overlap not resolved; preserve upstream channels as reported.")
                continue
            fields = text.split()
            try:
                if not selected:
                    raise ValueError("missing power header")
                stamp = parse_stamp(fields[0])
                power = sum(float(fields[i]) for i in selected) * factor
                if not math.isfinite(power):
                    raise ValueError("nonfinite power")
            except (ValueError, IndexError):
                malformed += 1
                evidence.add(rel, line_n, "unparsed_power_line", text)
                continue
            if first is None:
                first = stamp
                evidence.add(rel, line_n, "first_power_sample", text)
            if previous is not None:
                histogram[round((stamp - previous).total_seconds() * 1e6)] += 1
            previous = last = stamp
            count += 1
            minimum = min(minimum, power)
            maximum = max(maximum, power)
            second = stamp.replace(microsecond=0)
            sums[second] += power
            samples[second] += 1
            last_text = text
            last_line = line_n
    if last is not None:
        evidence.add(rel, last_line, "last_power_sample", last_text)
    coverage = dict(relative_path=rel, size_bytes=path.stat().st_size,
                    bytes_read=size, lines_read=line_n, sha256=h.hexdigest(),
                    eof_complete=size == path.stat().st_size, parser="power_stream",
                    records=count, malformed_lines=malformed, comment_lines=comments,
                    first_timestamp=first.isoformat() if first else None,
                    last_timestamp=last.isoformat() if last else None,
                    power_columns=columns, summed_power_columns=[columns[i] for i in selected],
                    min_reported_sum_w=minimum if count else None,
                    max_reported_sum_w=maximum if count else None,
                    timestamp_timezone="not declared; source wall-clock retained",
                    **_delta_summary(histogram))
    series = pd.DataFrame([{"timestamp": t, "mean_power_w": sums[t]/samples[t],
                            "sample_count": samples[t]} for t in sorted(sums)])
    return coverage, series


def scan_text(path: Path, raw_root: Path, year: int, evidence: Evidence):
    """Complete streaming scan of server/benchmark logs, including un-timed POSTs."""
    h = hashlib.sha256()
    size = 0
    counts = Counter()
    namespaces = []
    metrics = []
    rel = path.relative_to(raw_root).as_posix()
    line_n = 0
    with path.open("rb") as file:
        for line_n, raw in enumerate(file, 1):
            h.update(raw)
            size += len(raw)
            text = ANSI.sub("", raw.decode("utf-8", errors="replace")).strip()
            if text.startswith("Namespace("):
                fields = namespace_fields(text)
                namespaces.append(dict(fields, source_line=line_n, source_file=rel))
                counts["namespace_count"] += 1
                for key in ("ramp_up_strategy", "ramp_up_start_rps", "ramp_up_end_rps", "save_detailed", "save_result"):
                    if key in fields:
                        counts[f"{key}={fields[key]}"] += 1
                evidence.add(rel, line_n, "benchmark_namespace", text,
                             "Scalar configured offered rate; a Namespace is not a request-arrival record.")
            if "Starting initial single prompt test run" in text:
                counts["initial_single_prompt_test_count"] += 1
                evidence.add(rel, line_n, "initial_prompt_test", text,
                             "Per-benchmark setup/test confounds a direct old-rate to new-rate step.")
            if '"POST /v1/completions ' in text or '"POST /v1/chat/completions ' in text:
                counts["request_http_access_lines"] += 1
                timed = bool(STAMP.search(text))
                counts["timestamped_request_http_access_lines" if timed else "untimestamped_request_http_access_lines"] += 1
                evidence.add(rel, line_n, "timestamped_request_access" if timed else "untimestamped_request_access", text,
                             "Access/response log, not independently verified request-send time.")
            if "Avg prompt throughput:" in text:
                counts["server_metric_lines"] += 1
                m = METRICS.search(text)
                ts = SHORT_STAMP.search(text)
                if m and ts:
                    date = datetime.strptime(f"{year}-{ts.group(1)}", "%Y-%m-%d %H:%M:%S")
                    metrics.append(dict(timestamp=date, prompt_tokens_per_s=float(m[1]),
                                        generation_tokens_per_s=float(m[2]), running_requests=int(m[3]),
                                        waiting_requests=int(m[4]), source_line=line_n))
                    counts["parsed_server_metric_lines"] += 1
                    evidence.add(rel, line_n, "timestamped_server_metrics", text,
                                 "Coarse scheduler/throughput observation, not per-request arrival timestamps.")
                else:
                    counts["unparsed_server_metric_lines"] += 1
                    evidence.add(rel, line_n, "unparsed_server_metric", text)
            if any(word in text for word in ("Server is ready!", "Starting to load model", "Loading model weights", "Application startup complete", "Shutting down")):
                counts["startup_shutdown_markers"] += 1
                evidence.add(rel, line_n, "startup_shutdown", text)
            if any(key in text for key in REQUEST_DETAIL_KEYS):
                counts["request_detail_key_mentions"] += 1
                evidence.add(rel, line_n, "request_detail_key_mention", text)
    coverage = dict(relative_path=rel, size_bytes=path.stat().st_size, bytes_read=size,
                    lines_read=line_n, sha256=h.hexdigest(), eof_complete=size == path.stat().st_size,
                    parser="text_stream", **dict(counts))
    return coverage, pd.DataFrame(namespaces), pd.DataFrame(metrics)


def scan_json(path: Path, raw_root: Path, evidence: Evidence):
    """Read complete small metadata JSON/JSONL; inspect all nested key types."""
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    rel = path.relative_to(raw_root).as_posix()
    try:
        value = json.loads(text)
        records = value if isinstance(value, list) else [value]
        fmt = "json"
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        fmt = "jsonl"
    keys = Counter()
    list_keys = Counter()
    def visit(value):
        if isinstance(value, dict):
            for k, v in value.items():
                keys[k] += 1
                if isinstance(v, list):
                    list_keys[k] += 1
                visit(v)
        elif isinstance(value, list):
            for v in value:
                visit(v)
    visit(records)
    if fmt == "jsonl":
        if records:
            evidence.add(rel, 1, "aggregate_result_keys", text.splitlines()[0],
                         "All rows' keys audited; scalar latency summaries do not give per-request timestamps.")
    else:
        for number, line in enumerate(text.splitlines(), 1):
            if any(f'"{k}"' in line for k in ("request_rate", "start_time", "end_time", "burstiness", "batch_size", "id")):
                key = line.strip().split(":", 1)[0].strip('" ')
                evidence.add(rel, number, f"metadata_{key}", line)
    coverage = dict(relative_path=rel, size_bytes=path.stat().st_size, bytes_read=len(raw),
                    lines_read=len(text.splitlines()), sha256=hashlib.sha256(raw).hexdigest(),
                    eof_complete=True, parser=fmt, records=len(records), keys=dict(keys),
                    list_valued_keys=dict(list_keys),
                    request_detail_keys_present=sorted(REQUEST_DETAIL_KEYS.intersection(keys)))
    return coverage, value if fmt == "json" else records


def classify_gate(namespace_rows: pd.DataFrame, text_coverages: list[dict], has_power: bool,
                  has_run_boundaries: bool) -> dict:
    """Conservative A/B/C gate; absence claims refer only to audited fields."""
    dynamic_config = 0
    if not namespace_rows.empty:
        for key in ("ramp_up_strategy", "ramp_up_start_rps", "ramp_up_end_rps"):
            if key in namespace_rows:
                dynamic_config += int(namespace_rows[key].notna().sum())
    timed_access = sum(x.get("timestamped_request_http_access_lines", 0) for x in text_coverages)
    detail = sum(x.get("request_detail_key_mentions", 0) for x in text_coverages)
    return {
        "class_A_verified_rate_change_and_arrival_time": False,
        "class_B_start_stop_and_power_candidate": bool(has_power and has_run_boundaries),
        "class_C_fixed_rate_operating_points": bool(not namespace_rows.empty and "request_rate" in namespace_rows),
        "non_null_ramp_configuration_fields": dynamic_config,
        "timestamped_http_access_lines": timed_access,
        "request_detail_key_mentions": detail,
        "decision": "do_not_fit_or_select_transition_tau_from_this_audit",
        "reason": ("Run boundaries and power provide start/stop candidates, but do not identify an unconfounded "
                   "within-run request-rate step. Scalar rates/HTTP response lines cannot reconstruct exact "
                   "offered arrival times. Coarse scheduler metrics require separate causal/time alignment validation."),
        "gate_is_automatic_fit_authorization": False,
    }


def audit_dataset(raw_root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    evidence = Evidence()
    coverage = []
    family_results = {}
    for folder in sorted(raw_root.glob("inference*")):
        if not folder.is_dir():
            continue
        name = folder.name
        json_values = {}
        for path in sorted(folder.glob("*.json")):
            c, value = scan_json(path, raw_root, evidence)
            coverage.append(c)
            json_values[path.name] = value
        runs = json_values.get("log.json", [])
        year = parse_stamp(runs[0]["start_time"]).year if runs else 2026
        namespaces = []
        text_cov = []
        server_metrics = pd.DataFrame()
        for path in sorted(folder.glob("*.log")):
            if "wattameter" in path.name:
                continue
            c, n, m = scan_text(path, raw_root, year, evidence)
            coverage.append(c)
            text_cov.append(c)
            if not n.empty:
                namespaces.append(n)
            if not m.empty:
                server_metrics = m
        ns = pd.concat(namespaces, ignore_index=True) if namespaces else pd.DataFrame()
        if not ns.empty:
            ns.to_csv(output / f"{name}_benchmark_namespaces.csv", index=False)
        if not server_metrics.empty:
            server_metrics.to_csv(output / f"{name}_server_metrics.csv", index=False)
        streams = {}
        stream_cov = []
        for path in sorted(folder.glob("*wattameter*.log")):
            c, frame = scan_power(path, raw_root, evidence)
            coverage.append(c)
            stream_cov.append(c)
            sensor = "gpu" if "nvml" in path.name else "cpu_reported_domains"
            streams[sensor] = frame.rename(columns={"mean_power_w": f"{sensor}_w", "sample_count": f"{sensor}_samples"})
        joined = None
        for frame in streams.values():
            joined = frame if joined is None else joined.merge(frame, on="timestamp", how="outer", validate="one_to_one")
        if joined is not None:
            joined = joined.sort_values("timestamp")
            joined.to_csv(output / f"{name}_power_1s_diagnostic.csv", index=False)
        run_frame = pd.DataFrame(runs)
        if not run_frame.empty:
            run_frame["run_start"] = run_frame.start_time.map(parse_stamp)
            run_frame["run_end"] = run_frame.end_time.map(parse_stamp)
            run_frame["duration_s"] = (run_frame.run_end - run_frame.run_start).dt.total_seconds()
            run_frame["gap_after_previous_run_s"] = (run_frame.run_start - run_frame.run_end.shift()).dt.total_seconds()
            if "request_rate" in run_frame:
                run_frame["configured_rate_difference_from_previous_run"] = run_frame.request_rate.diff()
            if "id" in run_frame and not ns.empty:
                run_frame = run_frame.merge(ns[["id", "source_line", "source_file"]], on="id", how="left", validate="one_to_one")
            if stream_cov:
                first = max(parse_stamp(c["first_timestamp"]) for c in stream_cov)
                last = min(parse_stamp(c["last_timestamp"]) for c in stream_cov)
                run_frame["cpu_gpu_wallclock_span_covers_run"] = (run_frame.run_start >= first) & (run_frame.run_end <= last)
            if not server_metrics.empty:
                coarse_rows = []
                for _, row in run_frame.iterrows():
                    within = server_metrics.loc[(server_metrics.timestamp >= row.run_start) &
                                                (server_metrics.timestamp < row.run_end)]
                    active = within.loc[within.running_requests > 0]
                    coarse_rows.append({
                        "server_metric_rows_in_run": len(within),
                        "max_running_requests_coarse": int(within.running_requests.max()) if len(within) else None,
                        "max_waiting_requests_coarse": int(within.waiting_requests.max()) if len(within) else None,
                        "first_logged_active_after_benchmark_start_s":
                            (active.iloc[0].timestamp-row.run_start).total_seconds() if len(active) else None,
                    })
                run_frame = pd.concat([run_frame.reset_index(drop=True), pd.DataFrame(coarse_rows)], axis=1)
            if "id" in run_frame:
                result_records = json_values.get("log_vllm.json", [])
                result_ids = {str(row.get("id")) for row in result_records}
                run_frame["aggregate_result_available_in_local_log_vllm"] = run_frame.id.astype(str).isin(result_ids)
            run_frame["boundary_interpretation"] = "benchmark lifecycle boundary, not exact first/last request arrival"
            run_frame.to_csv(output / f"{name}_runs.csv", index=False)
        metric_dt = server_metrics.timestamp.diff().dt.total_seconds().dropna() if not server_metrics.empty else pd.Series(dtype=float)
        result = {
            "run_count": len(runs), "namespace_count": len(ns),
            "server_metrics_count": len(server_metrics),
            "server_metric_interval_median_s": float(metric_dt.median()) if len(metric_dt) else None,
            "server_metric_interval_min_s": float(metric_dt.min()) if len(metric_dt) else None,
            "server_metric_interval_max_s": float(metric_dt.max()) if len(metric_dt) else None,
            "run_duration_min_s": float(run_frame.duration_s.min()) if len(run_frame) else None,
            "run_duration_max_s": float(run_frame.duration_s.max()) if len(run_frame) else None,
            "request_rates": sorted(run_frame.request_rate.unique().tolist()) if "request_rate" in run_frame else [],
            "aggregate_result_records": len(json_values.get("log_vllm.json", [])),
            "aggregate_result_file_present": "log_vllm.json" in json_values,
            "power_valid_records": {c["relative_path"].split("/")[-1]: c["records"] for c in stream_cov},
            "power_rejected_records": {c["relative_path"].split("/")[-1]: c["malformed_lines"] for c in stream_cov},
            "run_wallclock_span_covered_by_both_power_streams_count":
                int(run_frame.cpu_gpu_wallclock_span_covers_run.sum()) if "cpu_gpu_wallclock_span_covers_run" in run_frame else 0,
            "gate": classify_gate(ns, text_cov, bool(stream_cov), bool(runs)),
        }
        if "aggregate_result_available_in_local_log_vllm" in run_frame and result["aggregate_result_file_present"]:
            result["run_ids_missing_aggregate_result"] = run_frame.loc[~run_frame.aggregate_result_available_in_local_log_vllm, "id"].tolist()
        if name.startswith("inference_offline"):
            result["gate"]["reason"] = "Offline finite batches have startup/drain boundaries, not online offered-rate step experiments."
        family_results[name] = result
        if joined is not None and len(run_frame):
            _plot_boundaries(name, run_frame, joined, server_metrics, output)
    (output / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(evidence.rows()).to_csv(output / "evidence_ledger.csv", index=False)
    gate = {
        "scope": "All files in the three local raw inference directories; all supplied .json and .log scanned to EOF.",
        "families": family_results,
        "files_scanned": len(coverage), "bytes_scanned": sum(c["bytes_read"] for c in coverage),
        "all_files_eof_complete": all(c["eof_complete"] for c in coverage),
        "raw_power_preserved": True, "raw_mutations": False,
        "power_diagnostic_resolution_s": 1,
        "no_fitted_tau": True,
        "timezone_and_clock": "Source wall-clock timestamps are naive. Span overlap is not proof of clock synchronization or causal lag.",
        "limits": [
            "No download or external benchmark harness code audit; completeness applies to local supplied files only.",
            "Per-run Namespace/config fixed-rate fields do not prove natural traffic or scheduler dynamics are absent.",
            "Adjacent runs can change rate on a persistent server, but initial test/setup/drain confound an instantaneous rate step.",
            "Access-log status lines are not validated arrival timestamps; preceding coarse metrics cannot timestamp each request.",
            "Diagnostic 1-second means are for visual inspection only, not reconstructed 1-second request-power causal pairs.",
            "CPU package/core domain overlap remains unresolved; figures show GPU separately to avoid claiming full-server boundary.",
        ],
        "recommended_next_action": "Keep instantaneous M3 as explicitly quasi-steady baseline. Do not add a claimed calibrated delay/first-order tau. Obtain time-aligned per-request send/completion and raw power around controlled warm-instance rate changes, or establish an externally justified processing-state model.",
    }
    (output / "audit_summary.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    return gate


def _plot_boundaries(name: str, runs: pd.DataFrame, power: pd.DataFrame,
                     metrics: pd.DataFrame, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Deterministic first three windows, not visually selected examples.
    selected = runs.iloc[:3]
    fig, axes = plt.subplots(len(selected), 1, figsize=(13, 3.2*len(selected)), squeeze=False)
    for ax, (_, run) in zip(axes[:, 0], selected.iterrows()):
        start, end = run.run_start, run.run_end
        left, right = start - pd.Timedelta(seconds=20), end + pd.Timedelta(seconds=20)
        p = power.loc[(power.timestamp >= left) & (power.timestamp < right)]
        ax.plot((p.timestamp-start).dt.total_seconds(), p.gpu_w/1000, label="GPU sum (1 s diagnostic mean)")
        ax.axvline(0, color="black", linestyle="--", label="Benchmark lifecycle boundary")
        ax.axvline((end-start).total_seconds(), color="black", linestyle="--")
        ax.set_ylabel("GPU power (kW)")
        rate = run.get("request_rate", "offline batch")
        ax.set_title(f"Run index {run.name}: configured rate {rate}; boundaries are NOT exact arrival steps")
        if not metrics.empty:
            m = metrics.loc[(metrics.timestamp >= left) & (metrics.timestamp < right)]
            ax2 = ax.twinx()
            ax2.plot((m.timestamp-start).dt.total_seconds(), m.running_requests, "o-", color="orange", alpha=.7, label="Running requests (coarse log)")
            ax2.set_ylabel("Running requests")
        ax.grid(alpha=.2)
        ax.set_xlabel("Time relative to benchmark start (s)")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle(name + " | raw coverage beyond the old trimmed 180 s window", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / f"{name}_boundary_diagnostics.png", dpi=160)
    plt.close(fig)
