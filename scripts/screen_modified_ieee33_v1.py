"""IEEE33-derived candidate audit, static AC screening and frozen-profile replay.

No OPF, adaptive DOE, infrastructure sizing optimization, or storage dispatch.
All bus labels in reports are original one-based IEEE labels.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import warnings
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandapower as pp
import pandapower.networks as pn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/modified_ieee33_candidate_v1"
CONFIG = ROOT / "configs/modified_ieee33_candidate_v1.json"


def read_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def original_network():
    # Parsing the packaged case is relatively expensive; never mutate this cache.
    return pn.case33bw()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(x):
    if isinstance(x, dict):
        return {str(k): serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [serial(v) for v in x]
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_json(path, obj):
    path.write_text(json.dumps(serial(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def build_network(cfg=None, *, original_voltage=False, background_scale=1.0,
                  branch_limit_ka=None, base_mva=None):
    cfg = cfg or read_config()
    spec = cfg["network"]
    net = copy.deepcopy(original_network())
    net.bus["name"] = [f"Bus {i+1}" for i in range(len(net.bus))]
    if not original_voltage:
        net.bus["vn_kv"] = spec["candidate_voltage_kv"]
    net.sn_mva = spec["calculation_base_mva"] if base_mva is None else base_mva
    net.bus["min_vm_pu"] = spec["voltage_min_pu"]
    net.bus["max_vm_pu"] = spec["voltage_max_pu"]
    net.line["max_i_ka"] = branch_limit_ka or spec["uniform_branch_limit_ka"]
    net.line["max_loading_percent"] = 100.0
    net.line["name"] = [f"Branch {int(r.from_bus)+1}-{int(r.to_bus)+1}" for _, r in net.line.iterrows()]
    net.load["scaling"] = background_scale
    net.ext_grid["vm_pu"] = spec["slack_voltage_pu"]
    # Legacy case generator box constraints are not physical transformer ratings.
    for col in ("max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar"):
        if col in net.ext_grid:
            net.ext_grid[col] = np.nan
    return net


def add_aidc(net, bus_label, p_mw, pf):
    assert 2 <= bus_label <= 33
    return pp.create_load(net, bus=bus_label-1, p_mw=p_mw,
                          q_mvar=p_mw*math.tan(math.acos(pf)), name="AIDC frozen PCC equivalent",
                          const_z_p_percent=0, const_i_p_percent=0,
                          const_z_q_percent=0, const_i_q_percent=0)


def solve(net, *, head_limit_mva=40.0, bus_label=None):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pp.runpp(net, algorithm="nr", init="flat", calculate_voltage_angles=True,
                     numba=False, max_iteration=40, tolerance_mva=1e-8,
                     check_connectivity=True)
    except pp.LoadflowNotConverged:
        return {"converged": False, "feasible": False, "failure": "AC solver did not converge; not a proof of physical voltage collapse"}
    v = net.res_bus.vm_pu
    if not np.isfinite(v).all():
        return {"converged": True, "feasible": False, "failure": "Unsupplied/nonfinite bus result"}
    lo = net.bus.index[v < net.bus.min_vm_pu-1e-7]
    hi = net.bus.index[v > net.bus.max_vm_pu+1e-7]
    active = net.line.index[net.line.in_service]
    overload = active[net.res_line.loc[active, "loading_percent"] > 100+1e-7]
    p = float(net.res_ext_grid.p_mw.sum())
    q = float(net.res_ext_grid.q_mvar.sum())
    s = math.hypot(p, q)
    losses = float(net.res_line.pl_mw.sum())
    r = {
        "converged": True,
        "feasible": len(lo)+len(hi)+len(overload)==0 and s <= head_limit_mva+1e-7,
        "min_voltage_pu": float(v.min()), "min_voltage_bus": int(v.idxmin())+1,
        "max_voltage_pu": float(v.max()),
        "under_voltage_buses": [int(i)+1 for i in lo], "over_voltage_buses": [int(i)+1 for i in hi],
        "max_line_current_ka": float(net.res_line.loc[active, "i_ka"].max()),
        "max_line_loading_percent": float(net.res_line.loc[active, "loading_percent"].max()),
        "overloaded_branches": net.line.loc[overload, "name"].tolist(),
        "head_p_mw": p, "head_q_mvar": q, "head_s_mva": s,
        "head_limit_mva": head_limit_mva, "head_overloaded": s > head_limit_mva+1e-7,
        "network_losses_mw": losses,
        "active_power_balance_error_mw": float(p-net.res_load.p_mw.sum()-losses),
    }
    if bus_label is not None:
        r["pcc_voltage_pu"] = float(v.loc[bus_label-1])
    return r


def snapshot(cfg, bus_label, p_mw, *, original_voltage=False, background=1.0, pf=None,
             current=None, head=None):
    pf = cfg["aidc"]["facility_lagging_power_factor"] if pf is None else pf
    net = build_network(cfg, original_voltage=original_voltage, background_scale=background, branch_limit_ka=current)
    add_aidc(net, bus_label, p_mw, pf)
    r = solve(net, head_limit_mva=head or cfg["network"]["feeder_head_apparent_power_limit_mva"], bus_label=bus_label)
    return {"bus": bus_label, "aidc_p_mw": p_mw, "aidc_q_mvar": p_mw*math.tan(math.acos(pf)),
            "voltage_kv": float(net.bus.vn_kv.iloc[0]), "background_scale": background, "pf": pf,
            "branch_limit_ka": float(net.line.max_i_ka.iloc[0]), **r}, net


def static_limit(cfg, bus_label):
    """Nominal single-period constant-PF loadability only, never dynamic DOE.

    Coarse scan checks that the observed feasible set is a prefix before bisection.
    Numerical nonconvergence ends the certified high-voltage solution branch; it
    does not certify absence of all other AC solutions.
    """
    upper = cfg["screening"]["static_p_search_upper_mw"]
    points = np.linspace(0, upper, 11)
    scan = [snapshot(cfg, bus_label, float(p))[0] for p in points]
    flags = [r["feasible"] for r in scan]
    if not flags[0]:
        return {"bus": bus_label, "baseline_feasible": False, "certified_p_mw": None, "coarse_flags": flags}
    if any(flags[j] and not flags[j-1] for j in range(1, len(flags))):
        return {"bus": bus_label, "baseline_feasible": True, "certified_p_mw": None, "nonmonotonic_scan": True, "coarse_flags": flags}
    if all(flags):
        return {"bus": bus_label, "baseline_feasible": True, "certified_p_mw": upper, "right_censored": True, "coarse_flags": flags}
    first = flags.index(False)
    low, high = float(points[first-1]), float(points[first])
    while high-low > cfg["screening"]["static_p_tolerance_mw"]:
        mid = (low+high)/2
        if snapshot(cfg, bus_label, mid)[0]["feasible"]:
            low = mid
        else:
            high = mid
    last = snapshot(cfg, bus_label, low)[0]
    violation = snapshot(cfg, bus_label, high)[0]
    if not violation["converged"]:
        boundary = "numerical_nonconvergence"
    else:
        boundary = []
        if violation["under_voltage_buses"] or violation["over_voltage_buses"]:
            boundary.append("voltage")
        if violation["overloaded_branches"]:
            boundary.append("line_current")
        if violation["head_overloaded"]:
            boundary.append("head_apparent_power")
    return {"bus": bus_label, "baseline_feasible": True, "certified_p_mw": low,
            "first_infeasible_p_mw": high, "boundary": boundary, "coarse_flags": flags,
            "last_feasible": last, "first_infeasible": violation}


def replay(cfg, profile):
    net = build_network(cfg)
    bus = cfg["aidc"]["base_case_connection_bus_1based"]
    pf = cfg["aidc"]["facility_lagging_power_factor"]
    idx = add_aidc(net, bus, float(profile[0]), pf)
    rows = []
    for k, p in enumerate(profile):
        net.load.at[idx, "p_mw"] = float(p)
        net.load.at[idx, "q_mvar"] = float(p)*math.tan(math.acos(pf))
        rows.append({"interval": k, "aidc_p_mw": float(p), **solve(net, bus_label=bus,
                      head_limit_mva=cfg["network"]["feeder_head_apparent_power_limit_mva"])})
    good = [r for r in rows if r["converged"] and "min_voltage_pu" in r]
    summary = {"bus": bus, "intervals": len(rows), "interval_seconds": 300,
               "duration_hours": len(rows)/12, "background": "Constant original nominal P/Q; no real-time background correlation",
               "all_converged": len(good)==len(rows), "violation_intervals": sum(not r["feasible"] for r in rows),
               "violation_hours": sum(not r["feasible"] for r in rows)/12,
               "min_voltage_pu": min(r["min_voltage_pu"] for r in good),
               "max_line_loading_percent": max(r["max_line_loading_percent"] for r in good),
               "max_head_s_mva": max(r["head_s_mva"] for r in good),
               "max_balance_error_mw": max(abs(r["active_power_balance_error_mw"]) for r in good)}
    return rows, summary


def main():
    cfg = read_config()
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [ROOT/cfg["aidc"]["profile"], ROOT/cfg["aidc"]["summary"],
                 ROOT/"configs/ess_tes_pcc_scheduling_v3_eta90.json"]
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    data = np.load(protected[0])
    profile = data[cfg["aidc"]["profile_key"]].copy()
    pmean, ppeak = float(profile.mean()), float(profile.max())
    raw = pn.case33bw()
    pp.to_json(raw, str(OUT/"original_case33bw.json"))
    origin = {"buses": len(raw.bus), "closed_branches": int(raw.line.in_service.sum()),
              "open_ties": int((~raw.line.in_service).sum()), "p_load_mw": float(raw.load.p_mw.sum()),
              "q_load_mvar": float(raw.load.q_mvar.sum()), "voltage_kv": float(raw.bus.vn_kv.iloc[0]),
              "raw_branch_max_i_ka": sorted(raw.line.max_i_ka.unique().tolist()),
              "raw_current_limit_note": "99999kA is pandapower's conversion placeholder for unspecified MATPOWER ratings. Not a real rating.",
              "source_sha256": digest(OUT/"original_case33bw.json")}
    native_net = copy.deepcopy(raw)
    native_result = solve(native_net, head_limit_mva=float("inf"))
    native_result["criteria"] = "Original MATPOWER0.9-1.1pu and no physical thermal rating; not comparable compliance verdict"
    strict_original = solve(build_network(cfg, original_voltage=True))
    base_result = solve(build_network(cfg))
    print("Original and modified no-AIDC baselines calculated", flush=True)
    labels = [cfg["aidc"]["base_case_connection_bus_1based"], *cfg["aidc"]["comparison_buses_1based"]]
    snapshots = []
    for original in (True, False):
        for bus in labels:
            for name, p in (("mean", pmean), ("peak", ppeak)):
                r, net = snapshot(cfg, bus, p, original_voltage=original)
                snapshots.append({"demand": name, **r})
                if not original and bus==labels[0] and name=="peak":
                    pp.to_json(net, str(OUT/"candidate_bus2_peak_network.json"))
    print("Location mean/peak snapshots calculated", flush=True)
    all_bus = [snapshot(cfg, b, ppeak)[0] for b in range(2, 34)]
    limits = [static_limit(cfg, b) for b in range(2, 34)]
    print("All32 non-slack buses and static loading boundaries calculated", flush=True)
    sensitivities = []
    for bus in labels:
        for bg in cfg["screening"]["background_scales"]:
            for pf in cfg["screening"]["facility_power_factors"]:
                sensitivities.append({"kind": "background_and_pf", **snapshot(cfg,bus,ppeak,background=bg,pf=pf)[0]})
        for i in cfg["screening"]["branch_limits_ka"]:
            for s in cfg["screening"]["head_limits_mva"]:
                sensitivities.append({"kind": "current_and_head_rating", **snapshot(cfg,bus,ppeak,current=i,head=s)[0]})
    rows, replay_summary = replay(cfg, profile)
    print("1438 interval baseline replay calculated", flush=True)
    write_json(OUT/"replay_5min.json", rows)
    write_json(OUT/"model_contract.json", cfg)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "status": cfg["status"], "config_sha256": digest(CONFIG),
              "versions": {p: importlib.metadata.version(p) for p in ("pandapower", "numpy")},
              "protected_input_hashes": before,
              "protected_inputs_unchanged": {k: digest(ROOT/k)==v for k,v in before.items()},
              "original": origin, "native_original_baseline": native_result,
              "original_voltage_under_study_criteria": strict_original, "modified_no_aidc_baseline": base_result,
              "aidc_mean_pcc_mw": pmean, "aidc_peak_pcc_mw": ppeak,
              "snapshots": snapshots, "all_bus_peak_screen": all_bus,
              "static_limits_not_doe": limits, "sensitivities": sensitivities,
              "baseline_replay": replay_summary,
              "main_bus_selected_from_peak_or_storage_results": False,
              "network_opf_or_doe_implemented": False, "storage_schedules_changed": False}
    write_json(OUT/"results.json", result)
    print(json.dumps(serial(replay_summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
