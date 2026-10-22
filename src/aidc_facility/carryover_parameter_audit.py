"""Phase 4-1: evidence and consistency checks, not a fitted queue simulator.

The conditional proxies here are not observed service times or validated capacity.
Do not pass this audit bundle to M3 as a new definition of its input rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .nlr_processing_diagnostic import SERVICES, WINDOWS, residence_screen

PRIMARY = "primary_60_240"
ANCHOR_RATES = (10., 20., 30.)


def validate_windows(frame: pd.DataFrame) -> None:
    required = {"run_id", "service", "configured_rps", "window", "running_mean",
        "running_max", "waiting_max", "unfinished_slope_rps", "unfinished_endpoint_slope_rps",
        "conditional_departure_rps", "metric_count", "max_metric_gap_s",
        "waiting_positive_all_snapshots", "low_drift_all_windows_2pct"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.duplicated(["run_id", "window"]).any():
        raise ValueError("Duplicate run/window key")
    if set(frame.service) != set(SERVICES):
        raise ValueError("Expected both NLR prompt datasets")
    numeric = list(required - {"run_id", "service", "window",
        "waiting_positive_all_snapshots", "low_drift_all_windows_2pct"})
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError("Non-finite required measurements")
    if (frame.configured_rps <= 0).any() or (frame.running_mean < 0).any():
        raise ValueError("Invalid rate or stock")
    for _, group in frame.groupby("run_id"):
        if len(group) != len(WINDOWS) or set(group.window) != set(WINDOWS):
            raise ValueError("Every run must have all four diagnostic windows")
        if group.service.nunique() != 1 or group.configured_rps.nunique() != 1:
            raise ValueError("Inconsistent run identity")
        passed = all(residence_screen(row) for row in group.to_dict("records"))
        if not group.low_drift_all_windows_2pct.eq(passed).all():
            raise ValueError("Stored residence screen disagrees with measurements")


def saturated_rows(windows: pd.DataFrame) -> pd.DataFrame:
    return windows[windows.configured_rps.ge(60)
        & windows.waiting_positive_all_snapshots.eq(True)
        & windows.metric_count.ge(6) & windows.max_metric_gap_s.le(20)].copy()


def slot_consistency(slots: float, lifetime_s: float, departure_rps: float) -> dict:
    """Little/rate identity for a hypothetical fully occupied constant-lifetime pool.

    This is an algebraic rejection diagnostic, not a measurement of NLR slots.
    """
    if not all(np.isfinite(v) and v > 0 for v in (slots, lifetime_s, departure_rps)):
        raise ValueError("Positive finite slots, lifetime and departure rate required")
    return {
        "hypothetical_slots": float(slots), "low_load_lifetime_proxy_s": float(lifetime_s),
        "implied_full_pool_rps": float(slots/lifetime_s),
        "conditional_high_load_departure_rps": float(departure_rps),
        "implied_to_conditional_ratio": float(slots/lifetime_s/departure_rps),
        "lifetime_needed_at_full_pool_s": float(slots/departure_rps),
        "effective_slots_needed_with_low_load_lifetime": float(departure_rps*lifetime_s),
    }


def build_audit(windows: pd.DataFrame, finite: pd.DataFrame,
                azure_tokens: pd.DataFrame, arrival_summary: dict) -> dict:
    validate_windows(windows)
    primary = windows[windows.window.eq(PRIMARY)]
    eligible = saturated_rows(windows)
    services = {}
    for service in SERVICES:
        p = primary[primary.service.eq(service)].sort_values("configured_rps")
        stable = p[p.low_drift_all_windows_2pct.eq(True)].copy()
        anchors = stable[stable.configured_rps.isin(ANCHOR_RATES)]
        if set(anchors.configured_rps) != set(ANCHOR_RATES):
            raise ValueError(f"Missing or unscreened 10/20/30 rps anchors: {service}")
        high = eligible[eligible.service.eq(service)]
        hp = high[high.window.eq(PRIMARY)]
        if hp.empty or (high.conditional_departure_rps <= 0).any():
            raise ValueError("Positive persistent-queue conditional departure estimates required")
        mu = float(hp.conditional_departure_rps.median())
        k_observed = float(p.running_max.max())
        w0 = float((anchors.running_mean/anchors.configured_rps).median())
        common_ids = set.intersection(*[
            set(high.loc[high.window.eq(name), "run_id"]) for name in WINDOWS])
        if not common_ids:
            raise ValueError("No common high-load cohort across windows")
        common = high[high.run_id.isin(common_ids)]
        sensitivity = []
        for name in WINDOWS:
            h = high[high.window.eq(name)]
            c = common[common.window.eq(name)]
            sensitivity.append({"window": name, "eligible_runs": len(h),
                "median_conditional_departure_rps": float(h.conditional_departure_rps.median()),
                "min_conditional_departure_rps": float(h.conditional_departure_rps.min()),
                "max_conditional_departure_rps": float(h.conditional_departure_rps.max()),
                "common_cohort_runs": len(c),
                "common_cohort_median_rps": float(c.conditional_departure_rps.median())})
        curve = []
        for _, row in stable.iterrows():
            g = windows[windows.run_id.eq(row.run_id)]
            values = g.running_mean/g.configured_rps
            curve.append({"run_id": row.run_id, "configured_rps": float(row.configured_rps),
                "running_mean": float(row.running_mean),
                "conditional_residence_proxy_s": float(row.running_mean/row.configured_rps),
                "window_min_s": float(values.min()), "window_max_s": float(values.max())})
        checks = [slot_consistency(k_observed, w0, mu), slot_consistency(64, w0, mu)]
        for item, name in zip(checks, ("NLR_observed_peak_as_hypothetical_slots", "PL43_64_slots_transfer_test")):
            item["case"] = name
        services[service] = {
            "conditional_departure": {"central_rps": mu,
                "window_envelope_min_rps": float(high.conditional_departure_rps.min()),
                "window_envelope_max_rps": float(high.conditional_departure_rps.max()),
                "status": "conditional_proxy_not_observed_capacity_or_SLO_limit",
                "sensitivity_is_confidence_interval": False, "windows": sensitivity},
            "concurrency": {"observed_peak_running": k_observed,
                "high_load_median_running": float(hp.running_mean.median()),
                "configured_slot_limit": None,
                "status": "observed_stock_not_verified_scheduler_setting"},
            "low_load_residence": {"anchor_rates_rps": list(ANCHOR_RATES),
                "anchor_median_s": w0,
                "primary_anchor_min_s": float((anchors.running_mean/anchors.configured_rps).min()),
                "primary_anchor_max_s": float((anchors.running_mean/anchors.configured_rps).max()),
                "status": "conditional_running_residence_not_measured_service_law",
                "curve": curve},
            "slot_consistency": checks,
            "selected_rate_evidence": p[p.configured_rps.isin([10,20,30,35,40,50,70,90])][[
                "run_id", "configured_rps", "running_mean", "running_max", "waiting_max",
                "unfinished_slope_rps", "conditional_departure_rps",
                "low_drift_all_windows_2pct"]].to_dict("records"),
            "pl43_lifetime_parameters": {"alpha0": None, "alpha1": None,
                "sigma_ttft": None, "mu_log_tbt": None, "sigma_log_tbt": None},
        }
    finite_summary = finite.groupby(["service", "hf-output-len"], as_index=False).agg(
        run_count=("id", "size"), mean_output_tokens=("observed_output_tokens_per_completed_request", "mean"),
        min_mean_e2el_s=("mean_e2el_s", "min"), max_mean_e2el_s=("mean_e2el_s", "max"))
    return {
        "schema_version": 1, "stage": "4-1: carryover parameter evidence and consistency audit",
        "parameter_evidence_audit_complete": True, "queue_model_calibrated": False,
        "power_mapping_validated": False, "facility_profile_generated": False,
        "services": services,
        "arrival_contract": {"reference": "facility_arrival_scaling_v1",
            "source_common_start_utc": arrival_summary["start_utc"],
            "source_common_end_exclusive_utc": arrival_summary["end_exclusive_utc"],
            "source_mix": arrival_summary["service_mix"],
            "multiplier_candidates_not_selected": arrival_summary["config"]["comparison_multipliers"],
            "facility_multiplier_selected": None, "instance_counts_selected": None,
            "mapping": {"code": SERVICES[0], "conversation": SERVICES[1]},
            "mapping_status": "service_class_analogy_not_token_or_latency_equivalence",
            "per_instance_routing_policy": None,
            "online_forecast_input": False},
        "token_transfer_evidence": {"azure_scope": "full available source traces, not common-window subset",
            "azure_statistics": azure_tokens.to_dict("records"),
            "nlr_finite_summary": finite_summary.to_dict("records"),
            "finite_is_online_rate_calibration": False,
            "token_conditioned_service_parameters_identifiable_from_these_summaries": False},
        "next_stage_controls": {
            "output_interval_s": 300, "candidate_internal_steps_s": [0.1, 0.5, 1.0],
            "internal_step_selected": None, "internal_steps_are_observations": False,
            "reset_at_5min_boundary": False, "baseline_other_exits": 0,
            "other_exits_status": "proposed_no_drop_baseline_assumption_not_observed_fact",
            "initial_state_policy": "observed_state_for_replay; warmup_and_sensitivity_for_synthetic",
            "queue_is_dispatch_flexibility": False,
            "R3_in_initial_state_validation": False,
            "M3_input_change_authorized_by_this_audit": False},
        "not_identified": ["actual scheduler admissions and per-request completions",
            "configured concurrency limit and exact scheduling policy",
            "load-dependent lifetime/remaining-work distribution",
            "separate TTFT, queue time, decode time and token-conditioned parameters",
            "causal processing-state to power mapping",
            "facility instance counts, routing, overload and timeout policy"],
    }
