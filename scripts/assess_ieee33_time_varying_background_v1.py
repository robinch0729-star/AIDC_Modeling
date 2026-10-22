"""Freeze the approved reference and assess explicitly declared background profiles.

No profile fitting to voltage results, storage rescheduling, DOE or new sizing.
SimBench contributes only a demand shape; IEEE33 is still the electrical network.
"""
from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp
import simbench as sb

import screen_ieee33_reinforcement_v1 as previous

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ieee33_time_varying_background_v1.json"
OUT = ROOT / "outputs/ieee33_time_varying_background_v1"
write_json = previous.core.write_json
digest = previous.core.digest


def read_config():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    ref = json.loads((ROOT / cfg["network_reference"]).read_text(encoding="utf-8"))
    return cfg, ref


def selected_candidate(ref, role):
    return next(c for c in previous.candidates(previous.read_config())
                if c["id"] == ref[role]["candidate_id"])


def build_network(ref, role):
    net = previous.build_network(previous.read_config(), selected_candidate(ref, role))
    line = net.line.iloc[0]
    expected = ref[role]
    actual = [line.r_ohm_per_km * line.length_km / line.parallel,
              line.x_ohm_per_km * line.length_km / line.parallel,
              line.c_nf_per_km * line.length_km * line.parallel,
              line.max_i_ka * line.parallel * line.df]
    wanted = [expected[k] for k in ("equivalent_root_r_ohm", "equivalent_root_x_ohm",
                                    "equivalent_root_c_nf", "root_current_limit_ka")]
    np.testing.assert_allclose(actual, wanted, rtol=0, atol=1e-10)
    assert float(net.ext_grid.vm_pu.iloc[0]) == ref["fixed"]["source_voltage_pu"]
    return net


def expand_and_trim(native, intervals, offset=1):
    values = np.asarray(native, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Invalid native profile")
    expanded = np.repeat(values, 3)
    if offset < 0 or offset + intervals > len(expanded):
        raise ValueError("Requested window outside source profile")
    return expanded[offset:offset + intervals].copy()


def make_scenarios(raw, cfg):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 1 or not np.isfinite(raw).all() or np.any(raw < 0) or raw.mean() <= 0:
        raise ValueError("Invalid shape")
    result = {}
    for spec in cfg["scenarios"]:
        divisor = {"mean": raw.mean, "max": raw.max}[spec["normalization"]]()
        result[spec["id"]] = np.roll(raw / divisor, spec["shift_steps"])
    return result


def make_background(cfg, ref, data):
    if importlib.metadata.version("simbench") != cfg["source"]["version"]:
        raise ValueError("SimBench version differs from declared source")
    donor = sb.get_simbench_net(cfg["source"]["profile_donor_code"])
    native = donor.profiles["load"]
    times = pd.to_datetime(native["time"], format="%d.%m.%Y %H:%M")
    # Annual naive labels contain spring/fall clock changes. Never silently
    # deduplicate or interpolate them; validate the selected May window instead.
    annual_step_anomalies = times.diff().dt.total_seconds().dropna()
    annual_step_anomalies = annual_step_anomalies[annual_step_anomalies != 900]
    loads = donor.load.loc[donor.load.in_service].copy()
    loads["weight"] = loads.p_mw * loads.scaling
    weights = loads.groupby("profile")["weight"].sum()
    cols = [f"{p}_pload" for p in weights.index]
    vals = native[cols].to_numpy(dtype=float)
    assert np.isfinite(vals).all() and (vals >= 0).all()
    aggregate = vals @ weights.to_numpy()
    # Verify the package's documented API produces the same aggregation.
    absolute = sb.get_absolute_profiles_from_relative_profiles(donor, "load", "p_mw")
    api = (absolute.loc[:, loads.index] * loads.scaling.to_numpy()).sum(axis=1).to_numpy()
    assert np.max(np.abs(api - aggregate)) < 1e-10
    start = pd.Timestamp(cfg["source"]["source_start_naive"])
    end = start + pd.Timedelta(days=cfg["source"]["source_days"])
    mask = ((times >= start) & (times < end)).to_numpy()
    source = aggregate[mask]
    assert len(source) == 480 and start.dayofweek == 6
    selected_times = times[mask]
    assert selected_times.is_unique and selected_times.is_monotonic_increasing
    assert (selected_times.diff().dropna().dt.total_seconds() == 900).all()
    n = cfg["alignment"]["expected_intervals"]
    assert len(data["no_storage_pcc_mw"]) == n
    raw = expand_and_trim(source, n, cfg["alignment"]["source_offset_s"] // 300)
    scenarios = make_scenarios(raw, cfg)
    stamps = pd.date_range(cfg["alignment"]["aidc_start_utc"], periods=n, freq="5min")
    old_meta = json.loads((ROOT / "outputs/pcc_cooling_tes_candidate_v1/summary.json").read_text(encoding="utf-8"))
    assert pd.Timestamp(old_meta["start_utc"]) == stamps[0]
    expected_hours = np.arange(n) / 12
    np.testing.assert_allclose(data["time_h"], expected_hours, atol=1e-10)
    frame = pd.DataFrame(dict(interval=np.arange(n), time_h=expected_hours,
                              aidc_timestamp_utc=stamps.astype(str),
                              raw_donor_aggregate_mw=raw))
    for name, scale in scenarios.items():
        frame[name + "_scale"] = scale
        frame[name + "_background_p_mw"] = scale * ref["fixed"]["nominal_background_p_mw"]
        frame[name + "_background_q_mvar"] = scale * ref["fixed"]["nominal_background_q_mvar"]
    frame.to_csv(OUT / "background_profiles_5min.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dict(source_timestamp_naive=times[mask].astype(str),
                      donor_aggregate_mw=source)).to_csv(OUT / "source_window_15min.csv", index=False)
    package = Path(sb.__file__).resolve().parent / cfg["source"]["dataset_folder"]
    audit = dict(
        source_package=importlib.metadata.version("simbench"),
        profile_donor_code=cfg["source"]["profile_donor_code"],
        source_files={f:dict(path=str(package / f), sha256=digest(package / f))
                      for f in ("LoadProfile.csv", "Load.csv")},
        native_rows=len(times), native_step_s=900,
        annual_duplicate_naive_labels=int(times.duplicated().sum()),
        annual_non15minute_steps=[dict(timestamp=str(times.loc[i]),step_s=float(v)) for i,v in annual_step_anomalies.items()],
        selected_window_duplicates=0, selected_window_non15minute_steps=0, missing_used_values=0,
        source_start_naive=str(start), source_end_exclusive_naive=str(end),
        profiles_weighted_mw=weights.to_dict(), aggregate_api_max_error_mw=float(np.max(np.abs(api-aggregate))),
        source_window_intervals=len(source), target_intervals=n,
        original_15min_energy_mwh=float(source.sum()/4),
        expanded_full_5min_energy_mwh=float(np.repeat(source,3).sum()/12),
        trimmed_5min_energy_mwh=float(raw.sum()/12),
        removed_edge_energy_mwh=float((source[0]+source[-1])/12),
        alignment=cfg["alignment"], scenarios={})
    for name, alpha in scenarios.items():
        p = alpha * ref["fixed"]["nominal_background_p_mw"]
        audit["scenarios"][name] = dict(min_scale=float(alpha.min()), mean_scale=float(alpha.mean()),
                max_scale=float(alpha.max()), p_min_mw=float(p.min()), p_mean_mw=float(p.mean()),
                p_max_mw=float(p.max()), energy_mwh=float(p.sum()/12),
                pearson_with_aidc_no_storage=float(np.corrcoef(p,data["no_storage_pcc_mw"])[0,1]))
    write_json(OUT / "profile_audit.json", audit)
    return frame, scenarios, audit


def solve_fast(net, head_limit):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pp.runpp(net, algorithm="bfsw", init="flat", numba=False,
                     max_iteration=200, tolerance_mva=1e-8)
    except pp.LoadflowNotConverged:
        return dict(converged=False, feasible=False, failure="BFSW nonconvergence; unresolved, not physical collapse")
    v = net.res_bus.vm_pu.to_numpy()
    if not np.isfinite(v).all():
        return dict(converged=True, feasible=False, failure="Nonfinite/unsupplied bus")
    low = np.where(v < net.bus.min_vm_pu.to_numpy()-1e-7)[0]
    high = np.where(v > net.bus.max_vm_pu.to_numpy()+1e-7)[0]
    active = net.line.in_service.to_numpy(dtype=bool)
    loading = net.res_line.loading_percent.to_numpy()[active]
    line_ids = net.line.index[active]
    over = line_ids[loading > 100+1e-7]
    p, q = float(net.res_ext_grid.p_mw.sum()), float(net.res_ext_grid.q_mvar.sum())
    apparent = math.hypot(p,q)
    losses = float(net.res_line.pl_mw.sum())
    return dict(converged=True, feasible=bool(not len(low) and not len(high) and not len(over) and apparent <= head_limit+1e-7),
        min_voltage_pu=float(v.min()), min_voltage_bus=int(v.argmin())+1,
        max_voltage_pu=float(v.max()), pcc_voltage_pu=float(v[1]),
        under_voltage_buses=(low+1).tolist(), over_voltage_buses=(high+1).tolist(),
        overloaded_branches=net.line.loc[over,"name"].tolist(),
        max_line_current_ka=float(net.res_line.i_ka.to_numpy()[active].max()),
        max_line_loading_percent=float(loading.max()), root_loading_percent=float(net.res_line.loading_percent.iloc[0]),
        head_p_mw=p, head_q_mvar=q, head_s_mva=apparent, head_overloaded=apparent > head_limit+1e-7,
        network_losses_mw=losses,
        active_power_balance_error_mw=p-float(net.res_load.p_mw.sum())-losses,
        reactive_power_balance_error_mvar=q-float(net.res_load.q_mvar.sum())-float(net.res_line.ql_mvar.sum()))


def apply_operating_point(net, bg_indices, aidc_index, alpha, p, q):
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("Invalid background scale")
    net.load.loc[bg_indices,"scaling"] = alpha
    net.load.at[aidc_index,"p_mw"] = p
    net.load.at[aidc_index,"q_mvar"] = q
    net.load.at[aidc_index,"scaling"] = 1.0


def violation_windows(rows, stamps):
    flag = np.array([not r["feasible"] for r in rows], dtype=bool)
    starts = np.where(flag & ~np.r_[False,flag[:-1]])[0]
    ends = np.where(flag & ~np.r_[flag[1:],False])[0]+1
    return [dict(start_interval=int(a), end_interval_exclusive=int(b),
                 start_utc=str(stamps[a]), end_exclusive_utc=str(pd.Timestamp(stamps[b-1])+pd.Timedelta(minutes=5)),
                 duration_h=float((b-a)/12),
                 buses=sorted({i for r in rows[a:b] for i in r.get("under_voltage_buses",[])}))
            for a,b in zip(starts,ends)]


def replay(ref, role, alpha, p, q, name, stamps):
    alpha,p,q = [np.asarray(x, dtype=float) for x in (alpha,p,q)]
    if alpha.shape != p.shape or p.shape != q.shape or p.ndim != 1:
        raise ValueError("Replay shapes do not match")
    if not all(np.isfinite(x).all() for x in (alpha,p,q)):
        raise ValueError("Nonfinite replay inputs")
    net = build_network(ref,role)
    bg = net.load.index.copy()
    idx = previous.core.add_aidc(net,2,0,ref["fixed"]["facility_lagging_pf"])
    rows, cache = [], {}
    for k,(a,pk,qk) in enumerate(zip(alpha,p,q)):
        key = (float(a),float(pk),float(qk))
        if key not in cache:
            apply_operating_point(net,bg,idx,*key)
            cache[key] = solve_fast(net,ref["fixed"]["head_limit_mva"])
        rows.append(dict(interval=k,time_h=k/12,aidc_timestamp_utc=str(stamps[k]),
            background_scale=float(a),background_p_mw=float(a*3.715),background_q_mvar=float(a*2.3),
            pcc_p_mw=float(pk),pcc_q_mvar=float(qk),**cache[key]))
        if k % 720 == 0:
            print(f"{name}: {k}/{len(p)}",flush=True)
    summary = previous.native.summarize(rows)
    valid = [r for r in rows if r.get("converged") and "failure" not in r]
    summary.update(max_voltage_pu=max((r["max_voltage_pu"] for r in valid),default=None),
        over_voltage_intervals=sum(bool(r["over_voltage_buses"]) for r in valid),
        max_reactive_balance_error_mvar=max((abs(r["reactive_power_balance_error_mvar"]) for r in valid),default=None),
        network_loss_energy_mwh=sum(r["network_losses_mw"] for r in valid)/12,
        worst_bus_voltage_deficit_pu_hours=sum(max(0,.95-r["min_voltage_pu"]) for r in valid)/12,
        under_voltage_bus_labels=sorted({i for r in valid for i in r["under_voltage_buses"]}),
        violation_windows=violation_windows(rows,stamps), file=f"{name}.json")
    # Independent solver at worst voltage/current/head, first/last, plus a failed point if any.
    selected = {0,len(rows)-1}
    for key,op in (("min_voltage_pu",min),("max_line_loading_percent",max),("head_s_mva",max)):
        if valid:
            selected.add(op(valid,key=lambda r:r[key])["interval"])
    unresolved = [r for r in rows if "failure" in r]
    if unresolved: selected.add(unresolved[0]["interval"])
    checks=[]
    for k in sorted(selected):
        apply_operating_point(net,bg,idx,float(alpha[k]),float(p[k]),float(q[k]))
        bfsw=solve_fast(net,ref["fixed"]["head_limit_mva"])
        v=net.res_bus.vm_pu.to_numpy().copy() if "failure" not in bfsw else None
        nr=previous.core.solve(net,head_limit_mva=ref["fixed"]["head_limit_mva"],bus_label=2)
        if v is not None and "failure" not in nr:
            diff=float(np.max(np.abs(v-net.res_bus.vm_pu.to_numpy())))
            assert diff < 1e-6 and bfsw["feasible"] == nr["feasible"]
        else: diff=None
        checks.append(dict(interval=k,max_voltage_difference_pu=diff,bfsw_feasible=bfsw["feasible"],nr_feasible=nr["feasible"]))
    summary["solver_crosschecks"] = checks
    write_json(OUT / f"{name}.json",rows)
    return rows,summary


def discharge_diagnostic(ref,role,alpha,p0,storage,rows):
    net=build_network(ref,role); bg=net.load.index.copy()
    idx=previous.core.add_aidc(net,2,0,.98)
    tan=math.tan(math.acos(ref["fixed"]["facility_lagging_pf"]))
    points=[]
    for original in rows:
        if original["feasible"]: continue
        k=original["interval"]
        p=float(p0[k]-storage["total_discharge_e_mw"])
        q=float((p0[k]-storage["tes_discharge_e_mw"])*tan)
        apply_operating_point(net,bg,idx,float(alpha[k]),p,q)
        points.append(dict(interval=k,pcc_p_mw=p,pcc_q_mvar=q,**solve_fast(net,ref["fixed"]["head_limit_mva"])))
    return dict(tested_violating_intervals=len(points),passes_instantaneous_test=sum(r["feasible"] for r in points),
        still_violates=sum(not r["feasible"] and "failure" not in r for r in points),
        unresolved=sum("failure" in r for r in points),
        note="Optimistic rated discharge without SOC/duration/recharge constraints; pass is not a feasible storage plan.", points=points)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg,ref=read_config()
    protected=[ROOT/p for p in ("configs/ieee33_reinforcement_v1.json","configs/pcc_tes_operating_candidate_v1.json",
        "configs/ess_tes_pcc_scheduling_v3_eta90.json")]
    for folder in ("outputs/ieee33_reinforcement_v1","outputs/ess_tes_pcc_scheduling_v3_eta90"):
        protected += [p for p in (ROOT/folder).rglob("*") if p.is_file()]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in protected}
    data,storage=previous.inputs()
    frame,scenarios,audit=make_background(cfg,ref,data)
    write_json(OUT/"model_contract.json",cfg); write_json(OUT/"adopted_network.json",ref)
    p0=data["no_storage_pcc_mw"]; prefix="constant_25_5_joint"
    pj=data[prefix+"_pcc_mw"]
    pf=ref["fixed"]["facility_lagging_pf"]
    q0=p0*math.tan(math.acos(pf))
    qj=previous.native.pcc_q(pj,data[prefix+"_ess_charge_mw"],data[prefix+"_ess_discharge_mw"],pf)
    modes={"no_storage":(p0,q0),"existing_joint":(pj,qj),"aidc_off_diagnostic":(np.zeros_like(p0),np.zeros_like(p0))}
    stamps=frame.aidc_timestamp_utc.tolist()
    evaluations={}
    prior=json.loads((previous.OUT/"results.json").read_text(encoding="utf-8"))
    for role in ("reference","comparison"):
        pp.to_json(build_network(ref,role),str(OUT/f"{role}_network.json"))
        evaluations[role]={}
        for case,alpha in scenarios.items():
            entry={"replays":{}}
            saved={}
            for mode,(p,q) in modes.items():
                rows,summary=replay(ref,role,alpha,p,q,f"{role}_{case}_{mode}",stamps)
                entry["replays"][mode]=summary; saved[mode]=rows
            background_failed=np.array([not r["feasible"] for r in saved["aidc_off_diagnostic"]])
            nostorage_failed=np.array([not r["feasible"] for r in saved["no_storage"]])
            entry["violation_attribution"]=dict(
                background_infeasible_intervals=int(background_failed.sum()),
                no_storage_infeasible_intervals=int(nostorage_failed.sum()),
                additional_no_storage_violations_with_background_feasible=int((nostorage_failed & ~background_failed).sum()),
                interpretation="Same time-varying background with AIDC removed; diagnosis only, not a separately reinforced baseline.")
            if case=="mean_preserved":
                diag=discharge_diagnostic(ref,role,alpha,p0,storage,saved["no_storage"])
                write_json(OUT/f"{role}_optimistic_discharge_diagnostic.json",diag)
                entry["optimistic_discharge_diagnostic"]={k:v for k,v in diag.items() if k!="points"}
            evaluations[role][case]=entry
        # Frozen constant-background histories are linked, not recalculated or overwritten.
        evaluations[role]["constant_background_historical"]={"source":"outputs/ieee33_reinforcement_v1/results.json",
            "replays":prior["evaluations"][ref[role]["historical_key"]]["replays"]}
    unchanged={p:digest(ROOT/p)==sha for p,sha in hashes.items()}
    assert all(unchanged.values())
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),status="reference_confirmed_time_varying_background_assessed",
        network_config_sha256=digest(ROOT/cfg["network_reference"]),profile_config_sha256=digest(CONFIG),
        intervals=len(p0),step_s=300,duration_h=len(p0)/12,background_profile_audit=audit,
        evaluations=evaluations,protected_input_hashes=hashes,protected_inputs_unchanged=unchanged,
        profile_rescaled=False,storage_resized=False,storage_reoptimized=False,
        network_parameters_retuned_after_profile_results=False,doe_implemented=False,
        software={p:importlib.metadata.version(p) for p in ("pandapower","simbench","numpy","pandas")})
    write_json(OUT/"results.json",result)
    for role in ("reference","comparison"):
        for case in scenarios:
            r=evaluations[role][case]["replays"]["no_storage"]
            print(role,case,r["min_voltage_pu"],r["converged_violation_intervals"],flush=True)


if __name__=="__main__":
    main()
