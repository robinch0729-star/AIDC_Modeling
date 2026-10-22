"""Audit 0.1/1/5/10 s arrival bins and shifted boundaries without fitting dynamics.

The same 180 s source sample, count, scale, fleet and calibrated M3/R3 are
retained. Unsupported cases are input diagnostics only, with no clipping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aidc_facility.arrival_trace import verify_saved_sample
from aidc_facility.time_varying_it import FixedService, simulate_case
from aidc_facility.traffic_bin_diagnostic import (
    aggregate_arrivals, describe_bins, project_bins, ramp_metrics, to_dispatch,
)
from nlr_data.final_it_profile_generator import FinalITProfileGenerator


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_plots(output, bins, profiles, stats):
    widths = [10., 5., 1., .1]
    colors = {10.: "#285a87", 5.: "#239378", 1.: "#d87625", .1: "#a94150"}
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    for ax, service in zip(axes, ["code", "conversation"]):
        for width in widths:
            p = bins.loc[(bins.service == service) & (bins.requested_bin_s == width) & (bins.phase_fraction == 0)]
            ax.stairs(p.node_rate_rps, np.r_[p.time_s, 180], label=f"{width:g} s bins", color=colors[width], alpha=.75, lw=.9 if width == .1 else 1.3, zorder=1 if width == .1 else 2)
        ax.axhline(100, color="black", ls="--", lw=.7, label="M3 support upper bound")
        ax.set_title(service + ": raw count variation is scaled, not measured facility utilization")
        ax.set_ylabel("Scenario requests/s per unit")
        ax.grid(alpha=.2)
        ax.legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Elapsed time (s)")
    fig.savefig(output / "arrival_resolution_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, sharey=True, constrained_layout=True)
    for ax, width in zip(axes, [10., 5., 1.]):
        for phase, style in [(0., "-"), (.5, "--")]:
            case = f"bin_{width:g}s_phase_{phase:g}"
            p = profiles.loc[(profiles.case == case) & (profiles.scenario_id == 0)]
            if p.empty:
                continue
            ax.plot(p.time_s, p.total_it_mw, style, lw=1, label=f"phase {phase:g} of bin ({width*phase:g} s)")
        ax.set_title(f"{width:g} s bins: same arrivals, different arbitrary aggregation boundaries")
        ax.set_ylabel("Reported IT power (MW)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Elapsed time (s)")
    fig.suptitle("Quasi-steady diagnostics only: no validated transient response", fontsize=13)
    fig.savefig(output / "bin_phase_power_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    for width in [10., 5., 1.]:
        p = profiles.loc[(profiles.requested_bin_s == width) & (profiles.phase_fraction == 0) & (profiles.scenario_id == 0)]
        if p.empty:
            continue
        axes[0].plot(p.time_s, p.total_it_mw, color=colors[width], lw=.9, label=f"{width:g} s bins")
        axes[1].plot(p.time_s.iloc[1:], np.abs(np.diff(p.total_it_mw))/.1, color=colors[width], lw=.8, label=f"{width:g} s bins")
    axes[0].set_ylabel("Reported IT power (MW)")
    axes[0].set_title("Same 180 s arrivals and 0.1 s output grid; unsupported 0.1 s input not simulated")
    axes[1].set_ylabel("Absolute 0.1 s ramp (MW/s)")
    axes[1].set_xlabel("Elapsed time (s)")
    axes[1].set_title("Raw ramp includes bin-edge artifacts, not a physical ramp benchmark")
    for ax in axes:
        ax.legend(ncol=3)
        ax.grid(alpha=.2)
    fig.savefig(output / "power_and_ramp_by_bin.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/traffic_bin_diagnostic_v1")
    parser.add_argument("--scenarios", type=int, default=5)
    parser.add_argument("--skip-residual", action="store_true", help="Input and deterministic M3 diagnostics only")
    args = parser.parse_args()
    if args.scenarios < 1:
        parser.error("Positive scenario count required")
    started = time.perf_counter()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    traffic_dir = ROOT / "data/raw/azure_inference_sample"
    _, provenance = verify_saved_sample(traffic_dir)
    config_path = ROOT / "outputs/time_varying_it_v1/run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    duration = float(provenance["duration_s"])
    if duration != 180:
        raise ValueError("This matched diagnostic expects the original 180 s sample")
    services = tuple(FixedService(**s) for s in config["services"])
    generator = FinalITProfileGenerator.from_benchmark_artifact(ROOT / "outputs/r3_high_rate_benchmark")
    scale = config["reference_input_rate_scale"]
    protected = [config_path, ROOT / "outputs/time_varying_it_v1/facility_profiles.csv",
                 ROOT / "outputs/heterogeneous_equivalent_facility_17mw/facility_profiles_long.csv",
                 ROOT / "outputs/r3_high_rate_benchmark/train_validation_m3_parameters.csv",
                 ROOT / "outputs/r3_high_rate_benchmark/train_validation_residual_parameters.csv",
                 traffic_dir / "provenance.json", *[traffic_dir / f"{s.key}_arrival_sample.csv" for s in services]]
    hashes = {str(p): sha256(p) for p in protected}
    timestamps = {}
    start = pd.Timestamp(provenance["start_in_source_clock"])
    for s in services:
        source = pd.read_csv(traffic_dir / f"{s.key}_arrival_sample.csv")
        stamp = pd.to_datetime(source.TIMESTAMP, format="mixed", utc=True)
        timestamps[s.key] = (stamp - start).dt.total_seconds().to_numpy()
    all_bins, all_metrics, power, power_stats, eligibility = [], [], [], [], []
    m3_series = {}
    for width in (10., 5., 1., .1):
        for phase in (0., .5):
            case = f"bin_{width:g}s_phase_{phase:g}"
            cases = []
            for service in services:
                part = aggregate_arrivals(timestamps[service.key], duration, width, phase)
                source_mean = len(timestamps[service.key]) / duration
                reference_rate = service.reference_rate_rps * scale
                part["node_rate_rps"] = part.source_rate_rps * reference_rate / source_mean
                part["offered_rps"] = part.node_rate_rps * service.node_count
                part["support_status"] = [generator.request_rate_status(service.prompt_dataset, float(r)) for r in part.node_rate_rps]
                part["m3_node_kw"] = np.nan
                valid = part.support_status.ne("unsupported_extrapolation")
                part.loc[valid, "m3_node_kw"] = generator.m3_parameters[service.prompt_dataset].predict(part.loc[valid, "node_rate_rps"].to_numpy())
                part["case"], part["service"], part["prompt_dataset"] = case, service.key, service.prompt_dataset
                part["requested_bin_s"], part["phase_fraction"], part["node_count"] = width, phase, service.node_count
                stats = {"case": case, "service": service.key, "requested_bin_s": width, "phase_fraction": phase, **describe_bins(part)}
                expected = reference_rate * service.node_count * duration
                stats["offered_count_conserved"] = bool(np.isclose(stats["scenario_offered_requests"], expected, rtol=1e-12))
                if not stats["offered_count_conserved"]:
                    raise AssertionError("Scenario request count changed")
                all_metrics.append(stats)
                cases.append(part)
            frame = pd.concat(cases, ignore_index=True)
            all_bins.append(frame)
            unsupported = bool(frame.support_status.eq("unsupported_extrapolation").any())
            aligned = all(np.allclose(p.bin_s/.1, np.rint(p.bin_s/.1)) for p in cases)
            allowed = not unsupported and aligned
            eligibility.append({"case": case, "supported": not unsupported, "grid_aligned": bool(aligned),
                                "simulated": allowed and not args.skip_residual,
                                "reason": "supported_quasi_steady_diagnostic" if allowed else "unsupported_input_or_grid; no clipping, no idle substitution"})
            if not allowed:
                print(f"{case}: input-only diagnostic; unsupported={unsupported}, aligned={aligned}", flush=True)
                continue
            total_m3 = np.zeros(int(round(duration/.1)))
            for service, part in zip(services, cases):
                values, boundary = project_bins(part, "m3_node_kw")
                total_m3 += values * service.node_count / 1000
            m3_series[case] = total_m3
            base = {"case": case, "requested_bin_s": width, "phase_fraction": phase}
            power_stats.append({**base, "component": "M3", "scenario_id": -1, **ramp_metrics(total_m3, boundary)})
            if args.skip_residual:
                p = pd.DataFrame({"time_s": np.arange(len(total_m3))*.1, "total_it_mw": total_m3,
                                  "m3_total_mw": total_m3, "residual_total_mw": 0., "scenario_id": -1,
                                  "traffic_boundary": boundary, "case": case})
            else:
                print(f"Simulating {case}: {args.scenarios} residual seeds", flush=True)
                p, _, audit = simulate_case(generator, to_dispatch(frame), services, case=case,
                    scenario_count=args.scenarios, seed=config["seed"], interval_s=.1, burn_in_s=config["burn_in_s"])
                audit.to_csv(output / f"{case}_node_checks.csv", index=False)
                for sid, sample in p.groupby("scenario_id"):
                    power_stats.append({**base, "component": "M3_R3", "scenario_id": int(sid),
                                        **ramp_metrics(sample.total_it_mw, sample.traffic_boundary)})
            p["requested_bin_s"], p["phase_fraction"] = width, phase
            power.append(p)
    bins, stats = pd.concat(all_bins, ignore_index=True), pd.DataFrame(all_metrics)
    profiles, scores = pd.concat(power, ignore_index=True), pd.DataFrame(power_stats)
    phase_rows = []
    for width in (10., 5., 1.):
        first, shifted = m3_series.get(f"bin_{width:g}s_phase_0"), m3_series.get(f"bin_{width:g}s_phase_0.5")
        if first is None or shifted is None:
            continue
        delta = first - shifted
        phase_rows.append({"requested_bin_s": width, "M3_phase_difference_rmse_mw": float(np.sqrt(np.mean(delta**2))),
                           "M3_phase_max_abs_difference_mw": float(np.max(np.abs(delta))),
                           "M3_phase_mean_difference_mw": float(delta.mean()),
                           "meaning": "same arrivals; arbitrary boundary sensitivity, NOT prediction error"})
    bins.to_csv(output / "arrival_bins.csv", index=False)
    stats.to_csv(output / "arrival_summary.csv", index=False)
    profiles.to_csv(output / "facility_profiles.csv", index=False)
    scores.to_csv(output / "power_metrics.csv", index=False)
    pd.DataFrame(phase_rows).to_csv(output / "phase_sensitivity.csv", index=False)
    pd.DataFrame(eligibility).to_csv(output / "case_eligibility.csv", index=False)
    if not args.skip_residual:
        save_plots(output, bins, profiles, stats)
    unchanged = all(sha256(Path(path)) == digest for path, digest in hashes.items())
    if not unchanged:
        raise AssertionError("Protected inputs or historical results changed")
    summary = {"duration_s": duration, "output_interval_s": .1, "bin_widths_s": [10, 5, 1, .1],
        "phase_fractions": [0, .5], "source_count": {s: len(v) for s, v in timestamps.items()},
        "all_request_counts_conserved": bool(stats.offered_count_conserved.all()),
        "output_scaling_or_smoothing": False, "clipping": False, "new_transient_model_fitted": False,
        "cases": eligibility, "historical_and_input_hashes_unchanged": unchanged,
        "protected_sha256": hashes, "run_seconds": time.perf_counter() - started,
        "scenario_count": args.scenarios if not args.skip_residual else 0,
        "seed": config["seed"], "source_provenance": provenance,
        "seed_caveat": "Same seed labels; old generator draws arrays in bin-dependent chunks, so innovations are NOT exactly paired across bin widths. Deterministic M3 comparison isolates binning.",
        "interpretation": "One observed arrival window reaggregated, not independent traffic validation; phase RMSE is sensitivity, not prediction accuracy."}
    (output / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(stats[["case", "service", "node_min_rps", "node_max_rps", "unsupported_time_fraction"]].to_string(index=False))
    print(f"Finished in {summary['run_seconds']:.1f} s; protected hashes unchanged={unchanged}", flush=True)


if __name__ == "__main__":
    main()
