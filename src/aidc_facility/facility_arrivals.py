"""Stage-2 arrival scaling, deliberately independent of power/capacity models.

Raw Azure timestamps are aggregated, never overwritten. Three *assumption*
cases share exactly the same service-specific five-minute request totals:
  replay: multiply the observed one-second counts by K;
  conditional_60s: preserve each minute count, redistribute arrivals in that minute;
  conditional_300s: preserve each five-minute count, redistribute in that block.

Redistribution uses a multinomial with equal second probabilities: the bin-count
law of a homogeneous Poisson process CONDITIONED ON its block count. It is not
a fitted or verified Poisson model of Azure or an AIDC. Fixing block totals also
means the bins are not unconditionally independent. See MIT Urban Operations
Research, section 2.12.3: https://web.mit.edu/urban_or_book/www/book/chapter2/2.12.3.html
Choosing the block size and using Azure counts for facility synthesis are OUR
scenario adaptations, not formulas taken from the NLR paper.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SERVICES = ("code", "conversation")
METHODS = ("replay", "conditional_60s", "conditional_300s")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_counts(counts):
    x = np.asarray(counts)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("Counts must be a nonempty one-dimensional array")
    if not np.issubdtype(x.dtype, np.integer) or np.any(x < 0):
        raise ValueError("Counts must be nonnegative integers")
    return x.astype(np.int64, copy=False)


def aggregate_counts(counts, interval_s, offset_s=0):
    """Left-closed complete bins. Drop edge fragments explicitly for phase tests."""
    x = validate_counts(counts)
    if not isinstance(interval_s, (int, np.integer)) or interval_s < 1:
        raise ValueError("interval_s must be a positive integer")
    if not isinstance(offset_s, (int, np.integer)) or not 0 <= offset_s < interval_s:
        raise ValueError("offset_s must be an integer in [0, interval_s)")
    n = (len(x) - offset_s) // interval_s
    if n < 1:
        raise ValueError("No complete bin")
    return x[offset_s:offset_s + n * interval_s].reshape(n, interval_s).sum(axis=1)


def scale_arrivals(counts, multiplier, method, seed=0):
    """Return 1-s facility counts. K is a diagnostic multiplier, NOT node count.

    Integer K is intentional in this comparison: exact counts need no rounding.
    Real-valued scale calibration, routing, processing and queues are out of scope.
    """
    x = validate_counts(counts)
    if not isinstance(multiplier, (int, np.integer)) or isinstance(multiplier, bool) or multiplier < 1:
        raise ValueError("multiplier must be a positive integer")
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if int(x.sum()) * int(multiplier) > np.iinfo(np.int64).max:
        raise OverflowError("Scaled count exceeds int64")
    if method == "replay":
        return x * multiplier
    block_s = 60 if method == "conditional_60s" else 300
    if len(x) % block_s:
        raise ValueError("Redistribution requires complete parent blocks")
    totals = aggregate_counts(x, block_s) * multiplier
    rng = np.random.default_rng(seed)
    probabilities = np.full(block_s, 1 / block_s)
    return np.concatenate([rng.multinomial(int(n), probabilities) for n in totals])


def relative_within_block_rms(counts, block_s=60):
    """Descriptive RMS deviation from each block mean, divided by GLOBAL mean."""
    x = validate_counts(counts)
    if len(x) % block_s:
        raise ValueError("Incomplete block")
    rows = x.reshape(-1, block_s).astype(float)
    denominator = float(rows.mean())
    if denominator == 0:
        return 0.0
    return float(np.sqrt(np.mean((rows - rows.mean(axis=1, keepdims=True))**2)) / denominator)


def arrival_metrics(counts, interval_s, offset_s=0):
    grouped = aggregate_counts(counts, interval_s, offset_s)
    rate = grouped / interval_s
    mean = float(rate.mean())
    change = np.abs(np.diff(rate))
    boundary = ((np.arange(1, len(rate)) * interval_s + offset_s) % 300 == 0)
    out = {
        "bin_s": interval_s, "phase_offset_s": offset_s,
        "complete_bins": len(rate), "retained_seconds": len(rate) * interval_s,
        "dropped_edge_seconds": len(counts) - len(rate) * interval_s,
        "request_count_in_retained_bins": int(grouped.sum()),
        "mean_rps": mean, "std_rps": float(rate.std()),
        "cv": float(rate.std()/mean) if mean else 0.,
        "p95_rps": float(np.quantile(rate, .95)),
        "p99_rps": float(np.quantile(rate, .99)), "max_rps": float(rate.max()),
        "peak_to_mean": float(rate.max()/mean) if mean else 0.,
        "adjacent_change_p95_rps": float(np.quantile(change, .95)) if len(change) else 0.,
        "adjacent_change_p95_pct_mean": float(np.quantile(change, .95)/mean*100) if len(change) and mean else 0.,
        "zero_bin_fraction": float(np.mean(grouped == 0)),
    }
    out["boundary_300s_change_p95_rps"] = float(np.quantile(change[boundary], .95)) if boundary.any() else None
    out["interior_change_p95_rps"] = float(np.quantile(change[~boundary], .95)) if (~boundary).any() else None
    return out


def build_source_seconds(raw_dir, audit_dir, output_dir, chunksize=500_000):
    """Recount real timestamps; reconcile every 5-min bin with stage 1.

    Full raw SHA256 is checked on every invocation. A cached count array may be
    reused only with matching raw digests and observation boundaries.
    """
    raw_dir, audit_dir, output_dir = map(Path, (raw_dir, audit_dir, output_dir))
    audit = json.loads((audit_dir / "audit_summary.json").read_text(encoding="utf-8"))
    start = pd.Timestamp(audit["common_window"]["start_inclusive_utc"])
    end = pd.Timestamp(audit["common_window"]["end_exclusive_utc"])
    length = int((end - start).total_seconds())
    if length % 300 or start.value % (300 * 10**9):
        raise ValueError("Common window must align with full five-minute bins")
    files = {"code": "AzureLLMInferenceTrace_code_1week.csv",
             "conversation": "AzureLLMInferenceTrace_conv_1week.csv"}
    signature = {"start_utc": start.isoformat(), "end_exclusive_utc": end.isoformat(), "raw_sha256": {}}
    for service in SERVICES:
        print(f"Checking full source digest: {service}", flush=True)
        actual = sha256(raw_dir / files[service])
        expected = next(s["raw_sha256"] for s in audit["services"] if s["service"] == service)
        if actual != expected:
            raise ValueError(f"Raw source digest changed: {service}")
        signature["raw_sha256"][service] = actual
    cache = output_dir / "source_arrivals_1s.npz"
    meta = output_dir / "source_arrivals_metadata.json"
    reusable = cache.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8")) == signature
    counts = {}
    reference = pd.read_csv(audit_dir / "common_5min_requests.csv")
    for service in SERVICES:
        if reusable:
            with np.load(cache, allow_pickle=False) as z:
                counts[service] = z[service]
        else:
            arr = np.zeros(length, dtype=np.int64)
            seen = 0
            for frame in pd.read_csv(raw_dir / files[service], usecols=["TIMESTAMP"], chunksize=chunksize):
                ts = pd.to_datetime(frame.TIMESTAMP, format="mixed", utc=True, errors="raise")
                ns = ts.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
                valid = (ns >= start.value) & (ns < end.value)
                pos = ((ns[valid] - start.value) // 10**9).astype(np.int64)
                if len(pos):
                    unique, freq = np.unique(pos, return_counts=True)
                    arr[unique] += freq
                seen += len(frame)
                if seen % (chunksize * 10) == 0:
                    print(f"{service}: read {seen:,} rows", flush=True)
            counts[service] = arr
        expected = reference.loc[reference.service.eq(service)].sort_values("bin_start_utc")
        timestamps = pd.to_datetime(expected.bin_start_utc, utc=True)
        if not np.array_equal(timestamps.astype("datetime64[ns, UTC]").astype("int64"), start.value + np.arange(length//300)*300*10**9):
            raise ValueError("Stage-1 timestamps do not align")
        if not np.array_equal(aggregate_counts(counts[service], 300), expected.request_count.to_numpy()):
            raise ValueError(f"Stage-1 count reconciliation failed: {service}")
    if not reusable:
        np.savez_compressed(cache, **counts)
        meta.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    return counts, signature, reference
