"""Independent saved-output checks for stage-2 offered arrival construction."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/facility_arrival_scaling_v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    result = json.loads((OUTPUT/"run_summary.json").read_text(encoding="utf-8"))
    stage1 = ROOT/"outputs/azure_long_trace_audit_v1"
    reference = pd.read_csv(stage1/"common_5min_requests.csv")
    metrics = pd.read_csv(OUTPUT/"comparison_metrics.csv")
    demand = pd.read_csv(OUTPUT/"facility_demand_5min.csv")
    start, end = pd.Timestamp(result["start_utc"]), pd.Timestamp(result["end_exclusive_utc"])
    checks = []
    def check(name, passed, **detail):
        checks.append({"check": name, "passed": bool(passed), **detail})
    source = np.load(OUTPUT/"source_arrivals_1s.npz", allow_pickle=False)
    for service in ("code", "conversation"):
        x = source[service]
        check(f"source_length:{service}", len(x) == int((end-start).total_seconds()))
        five = reference[reference.service.eq(service)].sort_values("bin_start_utc").request_count.to_numpy()
        check(f"source_vs_stage1_five_minute:{service}", np.array_equal(x.reshape(-1, 300).sum(axis=1), five))
        minute = pd.read_csv(stage1/f"{service}_minute_counts.csv")
        time_col = "minute_start" if "minute_start" in minute else minute.columns[0]
        t = pd.to_datetime(minute[time_col], utc=True)
        y = minute.loc[(t >= start) & (t < end), "request_count"].to_numpy()
        check(f"source_vs_stage1_every_minute:{service}", np.array_equal(x.reshape(-1, 60).sum(axis=1), y))
        for k in result["config"]["comparison_multipliers"]:
            d = demand[demand.service.eq(service) & demand.multiplier.eq(k)]
            check(f"demand_export_totals:{service}:{k}", np.array_equal(d.facility_offered_requests.to_numpy(), five*k))
            check(f"demand_export_rate:{service}:{k}", np.allclose(d.facility_offered_rps*300, five*k))
    primary = result["primary_multiplier"]
    for method in ("replay", "conditional_60s", "conditional_300s"):
        path = OUTPUT/f"facility_arrivals_{method}_K{primary}_seed{result['seeds'][0]}.npz"
        with np.load(path, allow_pickle=False) as generated:
            check(f"keys:{method}", set(generated.files) == {"code", "conversation"})
            for service in generated.files:
                x = generated[service]
                check(f"nonnegative_integers:{method}:{service}", x.dtype == np.int64 and np.all(x >= 0))
                target = source[service].reshape(-1, 300).sum(axis=1)*primary
                check(f"saved_five_minute_counts:{method}:{service}", np.array_equal(x.reshape(-1, 300).sum(axis=1), target))
                if method == "replay":
                    check(f"replay_each_second:{service}", np.array_equal(x, source[service]*primary))
                if method == "conditional_60s":
                    check(f"saved_minute_counts:{service}", np.array_equal(x.reshape(-1, 60).sum(axis=1), source[service].reshape(-1, 60).sum(axis=1)*primary))
                for dt in (1, 10, 60, 300):
                    rate = x.reshape(-1, dt).sum(axis=1)/dt
                    row = metrics[metrics.method.eq(method) & metrics.multiplier.eq(primary) &
                        metrics.seed.eq(result["seeds"][0]) & metrics.service.eq(service) &
                        metrics.bin_s.eq(dt) & metrics.phase_offset_s.eq(0)].iloc[0]
                    check(f"saved_metric_mean_max:{method}:{service}:{dt}",
                          np.isclose(rate.mean(), row.mean_rps) and np.isclose(rate.max(), row.max_rps))
    aligned = metrics[metrics.multiplier.eq(primary) & metrics.bin_s.eq(300) & metrics.phase_offset_s.eq(0)]
    for service in ("code", "conversation", "total"):
        subset = aligned[aligned.service.eq(service)]
        for field in ("mean_rps", "std_rps", "peak_to_mean", "adjacent_change_p95_rps"):
            check(f"same_five_minute_metric_all_seeds:{service}:{field}", np.allclose(subset[field], subset[field].iloc[0]))
    local = json.loads((OUTPUT/"verification.json").read_text(encoding="utf-8"))
    check("runner_checks", local["all_passed"])
    for name, expected in json.loads((ROOT/"outputs/time_varying_it_v1/run_config.json").read_text(encoding="utf-8"))["locked_parameters"].items():
        check("historical_parameter_lock:"+name, digest(ROOT/"outputs/r3_high_rate_benchmark"/name) == expected)
    hashes = {p.name: digest(p) for p in sorted(OUTPUT.iterdir()) if p.suffix in (".npz", ".csv", ".png", ".html")}
    out = {"all_passed": all(c["passed"] for c in checks), "checks": checks,
           "saved_output_sha256": hashes,
           "scope": "Independent arithmetic, export and historical lock checks; NOT physical validation"}
    (OUTPUT/"independent_verification.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_passed": out["all_passed"], "checks": len(checks),
                      "failures": [c for c in checks if not c["passed"]]}, indent=2))
    if not out["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
