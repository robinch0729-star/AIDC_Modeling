"""Native-voltage IEEE33 tests with unchanged AIDC and storage trajectories.

No network reinforcement, AIDC resizing, OPF, DOE or storage reoptimization.
Prior 33-kV results and facility inputs are hash-protected.
"""
from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandapower as pp

import screen_modified_ieee33_v1 as core

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ieee33_1266kv_screen_v1.json"
OUT = ROOT / "outputs/ieee33_1266kv_screen_v1"


def read_config():
    request = json.loads(CONFIG.read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / request["inherits"]).read_text(encoding="utf-8"))
    cfg["version"] = request["version"]
    cfg["status"] = "user_selected_12.66kv_33bus_other_parameters_provisional"
    cfg["family"] = "IEEE33 original-voltage topology with study-specific AIDC and rating assumptions"
    n = cfg["network"]
    n["candidate_voltage_kv"] = request["user_selected"]["nominal_voltage_kv"]
    a = request["retained_provisional_assumptions"]
    n.update(voltage_min_pu=a["voltage_bounds_pu"][0], voltage_max_pu=a["voltage_bounds_pu"][1],
             uniform_branch_limit_ka=a["branch_limit_ka"],
             feeder_head_apparent_power_limit_mva=a["head_limit_mva"],
             background_load_scale=a["background_load_scale"], slack_voltage_pu=a["slack_voltage_pu"])
    n["provenance"]["candidate_voltage"] = "User selected original Baran-Wu voltage 12.66 kV; Inha 2025 paper is supporting precedent."
    n["provenance"]["electrical_modification"] = "No voltage or physical series impedance modification; original 12.66 kV retained."
    cfg["aidc"]["base_case_connection_bus_1based"] = a["connection_bus_1based"]
    cfg["aidc"]["facility_lagging_power_factor"] = a["facility_lagging_power_factor"]
    cfg["screening"]["time_series_scope"] = request["scope"]
    cfg["references"] = [r for r in cfg["references"] if r["id"] != "R4"]
    cfg["references"].append({"id": "INHA2025", **request["reference"]})
    cfg["not_claimed"] = [
        "Physical feeder ratings, connection approval or Korean 22.9kV equivalence",
        "Network-aware storage optimization, multi-period DOE, DFL or HIL",
        "All other numeric assumptions approved by the user's voltage selection",
        "AC nonconvergence proves physical voltage collapse",
        "Baseline voltage violations were caused by the added AIDC"
    ]
    cfg["native_voltage_test"] = request
    return cfg


def pcc_q(p_net, p_charge, p_discharge, pf):
    """ESS has Q=0; PF belongs to the remaining facility, not net import."""
    return (np.asarray(p_net) - np.asarray(p_charge) + np.asarray(p_discharge)) * math.tan(math.acos(pf))


def summarize(rows):
    good = [r for r in rows if r["converged"] and "min_voltage_pu" in r]
    def extremum(key, op):
        return op(r[key] for r in good) if good else None
    count = len(rows)
    violations = sum(not r["feasible"] for r in good)
    failures = count - len(good)
    return {
        "intervals": count, "step_s": 300, "duration_h": count / 12,
        "converged_intervals": len(good), "unresolved_intervals": failures,
        "converged_violation_intervals": violations, "converged_violation_hours": violations / 12,
        "unresolved_hours": failures / 12,
        "feasible_intervals": sum(r["feasible"] for r in good),
        "under_voltage_intervals": sum(bool(r["under_voltage_buses"]) for r in good),
        "line_overload_intervals": sum(bool(r["overloaded_branches"]) for r in good),
        "head_overload_intervals": sum(r["head_overloaded"] for r in good),
        "min_voltage_pu": extremum("min_voltage_pu", min),
        "max_line_current_ka": extremum("max_line_current_ka", max),
        "max_line_loading_percent": extremum("max_line_loading_percent", max),
        "max_head_s_mva": extremum("head_s_mva", max),
        "max_balance_error_mw": max(abs(r["active_power_balance_error_mw"]) for r in good) if good else None,
        "pcc_mean_mw": float(np.mean([r["pcc_p_mw"] for r in rows])) if rows else None,
        "pcc_peak_mw": max(r["pcc_p_mw"] for r in rows) if rows else None,
        "pcc_min_mw": min(r["pcc_p_mw"] for r in rows) if rows else None,
        "minimum_pcc_voltage_pu": extremum("pcc_voltage_pu", min),
    }


def replay(cfg, p, q, name):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if p.ndim != 1 or q.ndim != 1 or not len(p) or p.shape != q.shape:
        raise ValueError("P and Q must be nonempty one-dimensional arrays of equal length")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("P and Q must contain only finite values")
    net = core.build_network(cfg)
    bus = cfg["aidc"]["base_case_connection_bus_1based"]
    idx = core.add_aidc(net, bus, float(p[0]), cfg["aidc"]["facility_lagging_power_factor"])
    rows = []
    for k, (pk, qk) in enumerate(zip(p, q)):
        net.load.at[idx, "p_mw"] = float(pk)
        net.load.at[idx, "q_mvar"] = float(qk)
        result = core.solve(net, head_limit_mva=cfg["network"]["feeder_head_apparent_power_limit_mva"], bus_label=bus)
        rows.append({"interval": k, "time_h": k / 12, "pcc_p_mw": float(pk), "pcc_q_mvar": float(qk), **result})
        if k % 360 == 0:
            print(f"{name}: {k}/{len(p)} operating points", flush=True)
    core.write_json(OUT / f"{name}_replay.json", rows)
    return rows, summarize(rows)


def main():
    cfg = read_config()
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [ROOT / cfg["native_voltage_test"]["inherits"],
                 ROOT / "configs/ess_tes_pcc_scheduling_v3_eta90.json",
                 ROOT / "configs/pcc_tes_operating_candidate_v1.json"]
    for folder in ("outputs/modified_ieee33_candidate_v1", "outputs/ess_tes_pcc_scheduling_v3_eta90"):
        protected.extend(p for p in (ROOT / folder).rglob("*") if p.is_file())
    before = {str(p.relative_to(ROOT)): core.digest(p) for p in protected}
    with np.load(ROOT / cfg["aidc"]["profile"]) as data:
        p0 = data["no_storage_pcc_mw"].copy()
        prefix = cfg["native_voltage_test"]["frozen_storage_schedule_prefix"]
        pj = data[prefix + "_pcc_mw"].copy()
        ch = data[prefix + "_ess_charge_mw"].copy()
        dis = data[prefix + "_ess_discharge_mw"].copy()
        hours = data["time_h"].copy()
    assert len(p0) == len(pj) == len(hours) and np.allclose(np.diff(hours), 1 / 12)
    pf = cfg["aidc"]["facility_lagging_power_factor"]
    q0, qj = pcc_q(p0, 0, 0, pf), pcc_q(pj, ch, dis, pf)
    background = core.build_network(cfg)
    baseline = core.solve(background, head_limit_mva=cfg["network"]["feeder_head_apparent_power_limit_mva"])
    baseline_voltage = background.res_bus.vm_pu.to_list()
    source_bounds_net = core.build_network(cfg)
    source_bounds_net.bus["min_vm_pu"], source_bounds_net.bus["max_vm_pu"] = .9, 1.1
    source_bounds_baseline = core.solve(source_bounds_net, head_limit_mva=cfg["network"]["feeder_head_apparent_power_limit_mva"])
    snapshots, voltage_profiles = [], {"no_aidc": baseline_voltage}
    for bus in (2, 6, 18):
        for demand, p in (("mean", float(p0.mean())), ("peak", float(p0.max()))):
            row, net = core.snapshot(cfg, bus, p)
            snapshots.append({"demand": demand, **row})
            if bus == 2 and row["converged"]:
                voltage_profiles[f"bus2_{demand}"] = net.res_bus.vm_pu.to_list()
                if demand == "peak":
                    pp.to_json(net, str(OUT / "bus2_peak_network.json"))
    all_bus = [core.snapshot(cfg, bus, float(p0.max()))[0] for bus in range(2, 34)]
    limits = [core.static_limit(cfg, bus) for bus in (2, 6, 18)]
    print("No-AIDC, mean/peak, all-bus screening calculated", flush=True)
    r0, s0 = replay(cfg, p0, q0, "no_storage")
    rj, sj = replay(cfg, pj, qj, "existing_joint")

    # Independent, necessary root-line power bound; ignore Q/losses for an
    # optimistic bound. This is not an AC hosting limit or multi-period DOE.
    storage = json.loads((ROOT / cfg["aidc"]["summary"]).read_text(encoding="utf-8"))
    tes_cfg = json.loads((ROOT / "configs/pcc_tes_operating_candidate_v1.json").read_text(encoding="utf-8"))
    max_reduction = storage["ess_config"]["discharge_limit_mw"] + storage["coefficients"]["reduction_mwe_per_mwth"] * tes_cfg["tes"]["discharge_limit_mw_th"]
    root_s_cap = math.sqrt(3) * cfg["network"]["candidate_voltage_kv"] * cfg["network"]["slack_voltage_pu"] * cfg["network"]["uniform_branch_limit_ka"]
    background_p = float(background.load.p_mw.sum())
    necessary_aidc_p_cap = root_s_cap - background_p
    optimistic_p = p0 - max_reduction
    bound = {
        "interpretation": "Necessary root-line bound at fixed slack voltage, ignoring Q and positive losses. Instantaneous maximum ESS/TES discharge ignores SOC and duration, so it is optimistic, not an achievable schedule.",
        "root_line_mva_bound": root_s_cap, "background_p_mw": background_p,
        "optimistic_aidc_active_power_upper_bound_mw": necessary_aidc_p_cap,
        "instantaneous_storage_reduction_upper_bound_mw": max_reduction,
        "optimistic_after_storage_pcc_min_mw": float(optimistic_p.min()),
        "optimistic_after_storage_pcc_peak_mw": float(optimistic_p.max()),
        "min_excess_over_necessary_bound_mw": float((optimistic_p - necessary_aidc_p_cap).min()),
        "intervals_failing_necessary_bound_even_with_max_discharge": int(np.sum(optimistic_p > necessary_aidc_p_cap)),
        "root_current_lower_bound_at_best_pcc_ka": float((optimistic_p.min() + background_p) / (math.sqrt(3) * cfg["network"]["candidate_voltage_kv"])),
    }
    # A converged peak point can be independently solved with radial BFSW.
    _, cross_net = core.snapshot(cfg, 2, float(p0.max()))
    nr_v = cross_net.res_bus.vm_pu.to_numpy().copy()
    pp.runpp(cross_net, algorithm="bfsw", init="flat", numba=False,
             max_iteration=200, tolerance_mva=1e-8)
    crosscheck = {"algorithm": "independent BFSW versus NR at bus2 peak", "converged": bool(cross_net.converged),
                  "maximum_voltage_difference_pu": float(np.max(np.abs(cross_net.res_bus.vm_pu.to_numpy() - nr_v)))}
    unchanged = {k: core.digest(ROOT / k) == v for k, v in before.items()}
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": core.digest(CONFIG),
        "status": cfg["status"], "network_voltage_kv": 12.66,
        "no_aidc_study_bounds": baseline, "no_aidc_source_voltage_bounds_diagnostic": source_bounds_baseline,
        "snapshots": snapshots, "all_non_slack_bus_peak_screen": all_bus,
        "static_limits_not_doe": limits, "voltage_profiles": voltage_profiles,
        "no_storage_replay": s0, "existing_joint_replay": sj,
        "necessary_power_bound": bound, "independent_solver_check": crosscheck,
        "protected_input_hashes": before, "protected_inputs_unchanged": unchanged,
        "q_boundary": "Q=(P_net-P_ESS_ch+P_ESS_dis)*tan(acos(.98)); ESS Q=0",
        "storage_reoptimized": False, "aidc_rescaled": False, "background_rescaled": False,
        "network_reinforced": False, "opf_or_doe_implemented": False,
        "comparison": cfg["native_voltage_test"]["comparison_note"],
    }
    core.write_json(OUT / "model_contract.json", cfg)
    core.write_json(OUT / "results.json", result)
    assert all(unchanged.values()) and crosscheck["maximum_voltage_difference_pu"] < 1e-6
    print(json.dumps({"no_storage": s0, "existing_joint": sj, "necessary_power_bound": bound}, indent=2), flush=True)


if __name__ == "__main__":
    main()
