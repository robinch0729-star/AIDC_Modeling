"""Independent count controls and preservation checks for stage-1 Azure audit."""
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aidc_facility.arrival_trace import verify_saved_sample
from aidc_facility.azure_trace_audit import sha256


def main():
    output = ROOT / "outputs/azure_long_trace_audit_v1"
    audit = json.loads((output / "audit_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads(Path(audit["manifest"]).read_text(encoding="utf-8"))
    checks = []
    for source in manifest["sources"]:
        service = source["service"]
        summary = next(s for s in audit["services"] if s["service"] == service)
        newlines, last = 0, b""
        with Path(source["path"]).open("rb") as stream:
            for block in iter(lambda: stream.read(8*1024**2), b""):
                newlines += block.count(b"\n")
                last = block[-1:]
        physical_rows = newlines + int(last != b"\n") - 1
        checks.append({"service": service, "check": "CSV record count vs independent binary line count",
                       "actual": summary["rows"], "expected": physical_rows, "passed": summary["rows"] == physical_rows})
        for field, stats in summary["token_statistics"].items():
            n = stats["n"] + summary["invalid_"+field]
            checks.append({"service": service, "check": field+" histogram count plus invalid count",
                           "actual": n, "expected": physical_rows, "passed": n == physical_rows})
    five = pd.read_csv(output / "common_5min_requests.csv")
    daily = pd.read_csv(output / "common_daily_requests.csv")
    for service, p in five.groupby("service"):
        a = int(p.request_count.sum())
        b = int(daily.loc[daily.service.eq(service), "request_count"].sum())
        checks.append({"service": service, "check": "five-minute counts reconcile to daily totals",
                       "actual": a, "expected": b, "passed": a == b})
    verify_saved_sample(ROOT / "data/raw/azure_inference_sample")
    checks.append({"service": "legacy", "check": "existing 180-second sample hashes and bins unchanged",
                   "actual": "verified", "expected": "verified", "passed": True})
    legacy = json.loads((ROOT / "data/raw/azure_inference_sample/provenance.json").read_text(encoding="utf-8"))
    for source in manifest["sources"]:
        old = legacy["sources"][source["service"]]
        with Path(source["path"]).open("rb") as stream:
            stream.seek(old["start_byte"])
            actual = hashlib.sha256(stream.read(old["downloaded_bytes"])).hexdigest()
        checks.append({"service": source["service"], "check": "legacy downloaded byte block matches full release file",
                       "actual": actual, "expected": old["prefix_sha256"], "passed": actual == old["prefix_sha256"]})
    config = json.loads((ROOT / "outputs/time_varying_it_v1/run_config.json").read_text(encoding="utf-8"))
    for filename, expected in config["locked_parameters"].items():
        actual = sha256(ROOT / "outputs/r3_high_rate_benchmark" / filename)
        checks.append({"service": "legacy", "check": filename+" unchanged", "actual": actual,
                       "expected": expected, "passed": actual == expected})
    pd.DataFrame(checks).to_csv(output / "verification_checks.csv", index=False)
    result = {"all_passed": all(c["passed"] for c in checks), "checks": checks,
              "scope": "Data integrity/counts and existing-artifact preservation, not physical facility validation"}
    (output / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_passed"]:
        raise SystemExit("Verification failed; do not report completion")


if __name__ == "__main__":
    main()
