"""Bounded sampling of official Azure 2024 LLM arrival traces.

Source: Azure/AzurePublicDataset, AzureLLMInferenceDataset2024.md (CC-BY).
These are invocation/token traces, NOT measured power or NLR workload traces.
No synthetic fallback, time compression, timestamp jitter or row subsampling.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import urllib.request
import uuid

import numpy as np
import pandas as pd

SOURCE_README = "https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md"
BASE = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/"
SOURCES = {"code": BASE + "AzureLLMInferenceTrace_code_1week.csv", "conversation": BASE + "AzureLLMInferenceTrace_conv_1week.csv"}


def parse_prefix(payload: bytes, *, skip_first_partial: bool = False) -> pd.DataFrame:
    if skip_first_partial:
        payload = b"TIMESTAMP,ContextTokens,GeneratedTokens\n" + payload.split(b"\n", 1)[1]
    # A byte prefix may end in the middle of a record; remove that record.
    end = payload.rfind(b"\n")
    if end < 0:
        raise ValueError("No complete CSV record in prefix")
    frame = pd.read_csv(io.BytesIO(payload[:end+1]))
    required = {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}
    if not required.issubset(frame):
        raise ValueError(f"Unexpected official source columns: {frame.columns.tolist()}")
    frame = frame[list(sorted(required))].copy()
    frame["TIMESTAMP"] = pd.to_datetime(frame.TIMESTAMP, format="mixed", errors="raise")
    if frame.empty or not frame.TIMESTAMP.is_monotonic_increasing:
        raise ValueError("Prefix must be nonempty and chronological; cannot certify a partial window otherwise")
    for col in ("ContextTokens", "GeneratedTokens"):
        values = pd.to_numeric(frame[col], errors="raise")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid token counts")
        frame[col] = values
    return frame


def common_window(frames: dict[str, pd.DataFrame], duration_s: float, bin_s: float) -> tuple[pd.Timestamp, pd.Timestamp]:
    if set(frames) != set(SOURCES):
        raise ValueError("Both official service traces are required")
    if duration_s <= 0 or bin_s <= 0 or not np.isclose(duration_s / bin_s, round(duration_s / bin_s)):
        raise ValueError("Positive duration must be an integer multiple of bin width")
    # Predetermined first common whole-second start, not a volatile-window search.
    start = max(frame.TIMESTAMP.iloc[0] for frame in frames.values()).ceil("s")
    end = start + pd.Timedelta(seconds=duration_s)
    if any(frame.TIMESTAMP.iloc[-1] < end for frame in frames.values()):
        raise ValueError("Downloaded prefix does not cover the common window")
    return start, end


def bin_arrivals(frames: dict[str, pd.DataFrame], start: pd.Timestamp, duration_s: float, bin_s: float) -> pd.DataFrame:
    if duration_s <= 0 or bin_s <= 0 or not np.isclose(duration_s / bin_s, round(duration_s / bin_s)):
        raise ValueError("Duration must be an integer multiple of bin width")
    bins = int(round(duration_s / bin_s))
    result = pd.DataFrame({"time_s": np.arange(bins) * bin_s, "bin_s": bin_s})
    end = start + pd.Timedelta(seconds=duration_s)
    for service, frame in frames.items():
        selected = frame.loc[(frame.TIMESTAMP >= start) & (frame.TIMESTAMP < end)]
        elapsed = (selected.TIMESTAMP - start).dt.total_seconds().to_numpy()
        indices = np.floor(elapsed / bin_s).astype(int)
        counts = np.bincount(indices, minlength=bins)
        if len(counts) != bins or counts.sum() != len(selected):
            raise AssertionError("Arrival bin conservation failed")
        result[service + "_count"] = counts
    return result


def fetch_range(url: str, start_byte: int, byte_limit: int) -> tuple[bytes, dict]:
    # Cache-busting avoids stale signed asset redirects, not access controls.
    request = urllib.request.Request(url + "?research_sample=" + uuid.uuid4().hex,
        headers={"Range": f"bytes={start_byte}-{start_byte+byte_limit-1}", "User-Agent": "AIDC-research-trace-sampler/1.0", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if start_byte > 0 and response.status != 206:
            raise RuntimeError("Server ignored byte range; refusing an unbounded whole-file download")
        payload = response.read(byte_limit)
        meta = {"http_status": response.status, "content_range": response.headers.get("Content-Range"),
                "response_content_length": response.headers.get("Content-Length"), "downloaded_bytes": len(payload),
                "start_byte": start_byte}
    return payload, meta


def prepare_sample(output_dir: Path, duration_s: float = 180.0, bin_s: float = 10.0, max_mib_per_service: int = 8) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames, payloads, source_meta = {}, {}, {}
    size = 256 * 1024
    maximum = max_mib_per_service * 1024 * 1024
    if maximum < size:
        raise ValueError("Maximum source prefix must be at least 256 KiB")
    cumulative = 0
    heads, totals = {}, {}
    lookup = []
    for service, url in SOURCES.items():
        payload, meta = fetch_range(url, 0, size)
        cumulative += len(payload)
        heads[service] = parse_prefix(payload)
        cr = meta.get("content_range")
        if not cr or "/" not in cr or cr.rsplit("/", 1)[1] == "*":
            raise RuntimeError("Source file size/range support unavailable; cannot sample matching source dates safely")
        totals[service] = int(cr.rsplit("/", 1)[1])
        lookup.append({"service": service, **meta})
    # Source Code and Conversation begin on DIFFERENT dates. Find the first
    # common source-clock window, never align their separate first days.
    start = max(f.TIMESTAMP.iloc[0] for f in heads.values()).ceil("s")
    end = start + pd.Timedelta(seconds=duration_s)
    offsets = {}
    for service, url in SOURCES.items():
        if heads[service].TIMESTAMP.iloc[-1] >= start:
            offsets[service] = 0
            continue
        lo, hi = 0, totals[service] - 1
        # Sorted-source assumption is explicitly recorded. Every inspected
        # block is checked; complete weekly global ordering is not asserted.
        while hi - lo > 65536:
            mid = (lo + hi) // 2
            payload, meta = fetch_range(url, mid, min(65536, totals[service] - mid))
            cumulative += len(payload)
            block = parse_prefix(payload, skip_first_partial=True)
            lookup.append({"service": service, **meta, "first_timestamp": block.TIMESTAMP.iloc[0].isoformat()})
            if block.TIMESTAMP.iloc[0] < start:
                lo = mid
            else:
                hi = mid
        offsets[service] = lo
    while size <= maximum:
        for service, url in SOURCES.items():
            payload, meta = fetch_range(url, offsets[service], size)
            cumulative += len(payload)
            frames[service] = parse_prefix(payload, skip_first_partial=offsets[service] > 0)
            payloads[service] = payload
            source_meta[service] = {"url": url, **meta,
                "prefix_sha256": hashlib.sha256(payload).hexdigest(),
                "prefix_first_timestamp": frames[service].TIMESTAMP.iloc[0].isoformat(),
                "prefix_last_timestamp": frames[service].TIMESTAMP.iloc[-1].isoformat(),
                "complete_prefix_rows": len(frames[service]),
                "downloaded_prefix_monotonic": True,
                "total_remote_file_bytes": totals[service]}
        if any(frame.TIMESTAMP.iloc[0] > start for frame in frames.values()):
            raise RuntimeError("Byte search skipped the chosen start; refusing an incomplete sample")
        if all(frame.TIMESTAMP.iloc[-1] >= end for frame in frames.values()):
            break
        size *= 2
    else:
        raise RuntimeError("Byte cap reached before the predetermined window was covered; no synthetic fallback")
    bins = bin_arrivals(frames, start, duration_s, bin_s)
    bins.to_csv(output_dir / "traffic_bins.csv", index=False)
    diagnostics = []
    for service, frame in frames.items():
        selected = frame.loc[(frame.TIMESTAMP >= start) & (frame.TIMESTAMP < end)].copy()
        selected.to_csv(output_dir / f"{service}_arrival_sample.csv", index=False)
        # Save exact downloaded prefix bytes for independent parsing/reproducibility.
        (output_dir / f"{service}_downloaded_prefix.csv.part").write_bytes(payloads[service])
        source_meta[service]["window_request_count"] = len(selected)
        source_meta[service]["window_sample_sha256"] = hashlib.sha256((output_dir / f"{service}_arrival_sample.csv").read_bytes()).hexdigest()
        diffs = selected.TIMESTAMP.diff().dt.total_seconds().dropna()
        positive = diffs[diffs > 0]
        source_meta[service]["minimum_positive_timestamp_difference_s"] = float(positive.min()) if len(positive) else None
        source_meta[service]["duplicate_timestamp_fraction"] = float(selected.TIMESTAMP.duplicated().mean())
        for col in ("ContextTokens", "GeneratedTokens"):
            source_meta[service][f"{col}_mean"] = float(selected[col].mean())
        for width in (0.1, 1.0, 10.0):
            if not np.isclose(duration_s / width, round(duration_s / width)):
                continue
            rates = bin_arrivals({service: frame}, start, duration_s, width)[service + "_count"].to_numpy() / width
            diagnostics.append({"service": service, "bin_s": width, "mean_rps": rates.mean(), "std_rps": rates.std(),
                                "min_rps": rates.min(), "max_rps": rates.max(), "zero_bin_fraction": np.mean(rates == 0)})
    pd.DataFrame(diagnostics).to_csv(output_dir / "traffic_resolution_audit.csv", index=False)
    provenance = {
        "dataset": "Azure LLM inference trace 2024", "source_readme": SOURCE_README,
        "license": "CC-BY; see official source license", "reference": "DynamoLLM, IEEE HPCA 2025, DOI 10.1109/HPCA61900.2025.00102",
        "window_selection_rule": "first whole-second timestamp not before either full source file start; binary byte search for common date; no volatility selection",
        "start_in_source_clock": start.isoformat(), "end_exclusive_in_source_clock": end.isoformat(),
        "source_timezone": "recorded +00:00 offset preserved; facility local timezone not inferred",
        "duration_s": duration_s, "bin_s": bin_s, "time_compression_applied": False,
        "synthetic_arrivals": False, "timestamp_jitter_applied": False,
        "window_coverage_basis": "chronological inspected byte blocks bracket common start/end; global ordering of entire weekly files was not audited",
        "partial_download": True, "cumulative_downloaded_bytes": cumulative,
        "sample_selection_is_population_representative": False,
        "measurement_type": "invocation timestamp and token counts, NOT electric power",
        "sources": source_meta, "range_lookup_log": lookup,
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir


def verify_saved_sample(directory: Path) -> tuple[pd.DataFrame, dict]:
    """Check hashes AND rebuild count bins before using cached traffic."""
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    frames = {}
    start = pd.Timestamp(provenance["start_in_source_clock"])
    end = pd.Timestamp(provenance["end_exclusive_in_source_clock"])
    for service in SOURCES:
        meta = provenance["sources"][service]
        sample_bytes = (directory / f"{service}_arrival_sample.csv").read_bytes()
        prefix_bytes = (directory / f"{service}_downloaded_prefix.csv.part").read_bytes()
        if hashlib.sha256(sample_bytes).hexdigest() != meta["window_sample_sha256"]:
            raise ValueError(f"Sample hash mismatch: {service}")
        if hashlib.sha256(prefix_bytes).hexdigest() != meta["prefix_sha256"]:
            raise ValueError(f"Source byte-range hash mismatch: {service}")
        sample = parse_prefix(sample_bytes)
        prefix = parse_prefix(prefix_bytes, skip_first_partial=meta["start_byte"] > 0)
        rebuilt = prefix.loc[(prefix.TIMESTAMP >= start) & (prefix.TIMESTAMP < end)].reset_index(drop=True)
        pd.testing.assert_frame_equal(sample.reset_index(drop=True), rebuilt, check_dtype=False)
        if len(sample) != meta["window_request_count"]:
            raise ValueError("Sample count disagrees with provenance")
        frames[service] = sample
    traffic = pd.read_csv(directory / "traffic_bins.csv")
    rebuilt_bins = bin_arrivals(frames, start, provenance["duration_s"], provenance["bin_s"])
    pd.testing.assert_frame_equal(traffic, rebuilt_bins, check_dtype=False)
    return traffic, provenance
