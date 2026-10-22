"""Independent arithmetic/raw-sample checks; does not import the diagnostic module."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import re

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/nlr_processing_diagnostic_v1"
RAW = ROOT.parent/"NLR_dataset/00_raw_datasets/inference_online_rate_llama3_70b"
OLD = ROOT/"outputs/nlr_transition_audit_v1"


def main():
    checks = []
    def check(name, passed, detail=None):
        checks.append(dict(check=name, passed=bool(passed), detail=detail))
        if not passed:
            raise AssertionError(name+": "+str(detail))
    def digest(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1048576), b""):
                h.update(b)
        return h.hexdigest()

    metrics = pd.read_csv(OUT/"raw_server_metrics.csv")
    metrics.timestamp = pd.to_datetime(metrics.timestamp)
    windows = pd.read_csv(OUT/"window_diagnostics.csv", dtype={"run_id":str})
    raw_runs = pd.DataFrame(json.loads((RAW/"log.json").read_text()))
    raw_runs["start"] = pd.to_datetime(raw_runs.start_time.str.replace("_", "T"))
    raw_runs = raw_runs.set_index("id")
    slopes, means, maximums, counts = [], [], [], []
    for row in windows.itertuples():
        start = raw_runs.loc[row.run_id, "start"]
        left, right = start+pd.Timedelta(seconds=row.window_start_s), start+pd.Timedelta(seconds=row.window_end_s)
        f = metrics[metrics.timestamp.between(left, right, inclusive="left")]
        x = (f.timestamp-start).dt.total_seconds().to_numpy()
        stock = (f.running_requests+f.waiting_requests).to_numpy()
        slopes.append(abs(float(np.polyfit(x, stock, 1)[0])-row.unfinished_slope_rps))
        weights = np.r_[np.diff(x), 0]/2 + np.r_[0, np.diff(x)]/2
        avg = np.average(f.running_requests, weights=weights)
        means.append(abs(avg-row.running_mean))
        maximums.append(int(f.waiting_requests.max()) == row.waiting_max)
        counts.append(len(f) == row.metric_count)
    check("all_800_slopes_independent_polyfit", max(slopes)<1e-10, {"max_abs_error":max(slopes)})
    check("all_800_running_means_independent_weights", max(means)<1e-10, {"max_abs_error":max(means)})
    check("all_800_waiting_maxima", all(maximums))
    check("all_800_window_counts", all(counts))
    check("residence_proxy_not_reported_for_failed_all_window_screen",
          windows.loc[~windows.low_drift_all_windows_2pct, "conditional_residence_proxy_s"].isna().all())
    check("scheduler_stocks_nonnegative", (metrics[["running_requests","waiting_requests"]]>=0).all().all())
    check("metric_order_unique", metrics.timestamp.is_monotonic_increasing and not metrics.timestamp.duplicated().any())
    # Direct random-spaced/raw boundary samples, independent of the parser module.
    chosen = set(metrics.iloc[::137].source_line.astype(int))
    chosen.update(windows[windows.configured_rps.isin([40,90])].first_source_line.astype(int))
    wanted = metrics[metrics.source_line.isin(chosen)].set_index("source_line")
    matched = 0
    with (RAW/"vllm_server.log").open("rb") as f:
        for number, raw in enumerate(f, 1):
            if number not in chosen:
                continue
            line = raw.decode("utf-8", "replace")
            r = re.search(r"Running: (\d+) reqs, Waiting: (\d+) reqs", line)
            t = re.search(r"generation throughput: ([0-9.]+) tokens/s", line)
            expected = wanted.loc[number]
            check("raw_metric_line:"+str(number), r is not None and t is not None and
                  int(r[1])==expected.running_requests and int(r[2])==expected.waiting_requests and
                  float(t[1])==expected.generation_tokens_per_s)
            matched += 1
    check("all_selected_raw_lines_found", matched==len(chosen), {"raw_lines_checked":matched})
    # Independently reconstruct 36 selected seconds directly from the raw GPU stream.
    selected_seconds = set()
    for r in raw_runs[raw_runs.request_rate.isin([10,30,40,50,90,100])].itertuples():
        for dt in [65,155,235]:
            selected_seconds.add((r.start+pd.Timedelta(seconds=dt)).floor("s").strftime("%Y-%m-%d %H:%M:%S"))
    sums, n = defaultdict(float), defaultdict(int)
    gpu = next(RAW.glob("nvml_wattameter*.log"))
    with gpu.open("r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            second = parts[0][:19].replace("_", " ")
            if second in selected_seconds:
                sums[second] += sum(map(float, parts[2:6]))/1000
                n[second] += 1
    p = pd.read_csv(OLD/"inference_online_rate_llama3_70b_power_1s_diagnostic.csv").set_index("timestamp")
    check("36_raw_gpu_seconds_found", len(n)==36)
    check("raw_gpu_sample_counts", all(n[s]==p.loc[s,"gpu_samples"] for s in n))
    check("raw_gpu_mean_reconstruction", all(abs(sums[s]/n[s]-p.loc[s,"gpu_w"])<1e-8 for s in n))
    aligned = pd.read_csv(OUT/"aligned_logger_power.csv")
    check("aligned_power_duration", aligned.power_seconds.eq(aligned.interval_s).all())
    for service in ["InstructCoder", "mlperf-inference-llama2-data"]:
        group = windows[(windows.service.eq(service)) & windows.window.eq("primary_60_240")]
        high = group[group.configured_rps.ge(81)]
        check("all_high_rate_waiting:"+service, len(high)==20 and high.waiting_max.gt(0).all())
        r40 = group[group.configured_rps.eq(40)].iloc[0]
        check("40rps_running_growth_without_waiting:"+service,
              r40.waiting_max==0 and r40.running_slope_rps>.5 and not r40.low_drift_all_windows_2pct)
    finite = pd.read_csv(OUT/"finite_results.csv")
    check("finite_results_count", len(finite)==1024)
    check("finite_duration_identity", np.allclose(finite.completed/finite.duration, finite.request_throughput))
    for item in json.loads((OUT/"source_manifest.json").read_text()):
        check("source_hash_recheck:"+Path(item["path"]).name, digest(item["path"])==item["sha256"])
    for filename, expected in json.loads((ROOT/"outputs/time_varying_it_v1/run_config.json").read_text())["locked_parameters"].items():
        check("locked_model:"+filename, digest(ROOT/"outputs/r3_high_rate_benchmark"/filename)==expected)
    doc = (OUT/"report_ko.html").read_text(encoding="utf-8")
    for link in re.findall(r'(?:href|src)="([^"]+)"', doc):
        if not link.startswith(("http", "#")) and link!="independent_verification.json":
            check("report_local_link:"+link, (OUT/link).is_file())
    for path in sorted(OUT.glob("*.png")):
        with Image.open(path) as im:
            check("figure_integrity:"+path.name, im.width==1650 and im.height>=900)
            im.verify()
    result = dict(passed=all(c["passed"] for c in checks), count=len(checks), checks=checks,
                  note="Independent numerical/raw-file checks; not external physical validation")
    (OUT/"independent_verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(passed=result["passed"], checks=result["count"], raw_metrics_checked=matched,
                          raw_gpu_seconds_checked=len(n), windows_recalculated=len(windows)), indent=2))


if __name__ == "__main__":
    main()
