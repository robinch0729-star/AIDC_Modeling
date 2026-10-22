"""Finite root-corridor reinforcement screen with frozen facility/storage data.

Only research equivalents, not a physical cable/transformer design. Candidate
family/ranking are committed in config before selection. No dispatch fitting.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandapower as pp

import screen_ieee33_1266kv_v1 as native

core = native.core
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ieee33_reinforcement_v1.json"
OUT = ROOT / "outputs/ieee33_reinforcement_v1"


def read_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def candidates(cfg):
    taps = cfg["search"]["source_voltage_candidates_pu"]
    result = [dict(id=f"original_v{v:.3f}", circuits=1, area_mm2=0,
                   source_voltage_pu=v, replaced_corridors=0) for v in taps]
    for n, cable, v in itertools.product(cfg["search"]["parallel_circuit_candidates"], cfg["cables"], taps):
        result.append(dict(id=f"cu{cable['area_mm2']}_n{n}_v{v:.3f}", circuits=n,
                           area_mm2=cable["area_mm2"], source_voltage_pu=v, replaced_corridors=1))
    return result


def rank(candidate):
    return (candidate["replaced_corridors"], candidate["circuits"],
            candidate["area_mm2"], candidate["source_voltage_pu"])


def build_network(cfg, candidate, *, background=1.0, length=None, current_factor=1.0,
                  resistance_factor=1.0, other_current=None, source_offset=0.0):
    if current_factor <= 0 or resistance_factor <= 0 or background < 0:
        raise ValueError("Invalid current/resistance/background factor")
    net = core.build_network(native.read_config(), background_scale=background)
    fixed = cfg["fixed"]
    net.bus["vn_kv"] = fixed["nominal_voltage_kv"]
    net.bus["min_vm_pu"], net.bus["max_vm_pu"] = fixed["voltage_bounds_pu"]
    net.f_hz = fixed["frequency_hz"]
    net.ext_grid["vm_pu"] = candidate["source_voltage_pu"] + source_offset
    net.line["max_i_ka"] = fixed["other_branch_limit_ka"] if other_current is None else other_current
    roots = net.line.index[(net.line.from_bus == 0) & (net.line.to_bus == 1)]
    assert len(roots) == 1
    root = roots[0]
    if candidate["area_mm2"]:
        cable = next(c for c in cfg["cables"] if c["area_mm2"] == candidate["area_mm2"])
        route = cfg["search"]["assumed_route_length_km"] if length is None else length
        if route <= 0:
            raise ValueError("Route length must be positive")
        # Use per-circuit values with pandapower.parallel; do NOT divide R/X twice.
        values = dict(length_km=route, r_ohm_per_km=cable["r_ohm_per_km"] * resistance_factor,
                      x_ohm_per_km=cable["x_ohm_per_km_at_50hz"] * net.f_hz / 50,
                      c_nf_per_km=cable["c_nf_per_km"], g_us_per_km=0.0,
                      max_i_ka=cable["buried_trefoil_current_ka"], parallel=candidate["circuits"],
                      df=current_factor, type="cs")
        for key, value in values.items():
            net.line.at[root, key] = value
    else:
        net.line.at[root, "df"] = current_factor
        net.line.at[root, "r_ohm_per_km"] *= resistance_factor
    return net


def solve(net, cfg, head=None):
    result = core.solve(net, head_limit_mva=cfg["fixed"]["head_limit_mva"] if head is None else head, bus_label=2)
    if not result.get("converged") or "failure" in result:
        return result
    root = net.line.index[(net.line.from_bus == 0) & (net.line.to_bus == 1)][0]
    n = int(net.line.at[root, "parallel"])
    total = float(net.res_line.at[root, "i_ka"])
    other = net.line.index[net.line.in_service & (net.line.index != root)]
    result.update(root_total_current_ka=total, root_current_per_circuit_ka=total / n,
                  root_circuits=n, root_effective_total_limit_ka=float(n * net.line.at[root, "max_i_ka"] * net.line.at[root, "df"]),
                  root_loading_percent=float(net.res_line.at[root, "loading_percent"]),
                  other_max_loading_percent=float(net.res_line.loc[other, "loading_percent"].max()),
                  reactive_power_balance_error_mvar=float(net.res_ext_grid.q_mvar.sum() - net.res_load.q_mvar.sum() - net.res_line.ql_mvar.sum()))
    return result


def snapshot(cfg, candidate, p, q, *, head=None, **kwargs):
    net = build_network(cfg, candidate, **kwargs)
    idx = core.add_aidc(net, 2, p, cfg["fixed"]["facility_lagging_pf"])
    net.load.at[idx, "q_mvar"] = q
    return {"pcc_p_mw": p, "pcc_q_mvar": q, **solve(net, cfg, head)}, net


def inputs():
    base = native.read_config()
    with np.load(ROOT / base["aidc"]["profile"]) as data:
        d = {k: data[k].copy() for k in data.files}
    storage = json.loads((ROOT / base["aidc"]["summary"]).read_text(encoding="utf-8"))
    tes = json.loads((ROOT / "configs/pcc_tes_operating_candidate_v1.json").read_text(encoding="utf-8"))["tes"]
    e = storage["ess_config"]
    delta_ch_tes = storage["coefficients"]["increase_mwe_per_mwth"] * tes["charge_limit_mw_th"]
    delta_dis_tes = storage["coefficients"]["reduction_mwe_per_mwth"] * tes["discharge_limit_mw_th"]
    return d, dict(ess=e, tes=tes, tes_charge_e_mw=delta_ch_tes, tes_discharge_e_mw=delta_dis_tes,
                   total_charge_e_mw=e["charge_limit_mw"]+delta_ch_tes,
                   total_discharge_e_mw=e["discharge_limit_mw"]+delta_dis_tes)


def operating_points(cfg, profile, storage):
    tan = math.tan(math.acos(cfg["fixed"]["facility_lagging_pf"]))
    lo, mean, hi = float(profile.min()), float(profile.mean()), float(profile.max())
    pts = [dict(name=name, p=p, q=p*tan, background=1.0)
           for name, p in (("aidc_min",lo), ("aidc_mean",mean), ("aidc_peak",hi))]
    pts.extend([
        dict(name="peak_plus_rated_charge_envelope", p=hi+storage["total_charge_e_mw"],
             q=(hi+storage["tes_charge_e_mw"])*tan, background=1.0),
        dict(name="min_minus_rated_discharge_envelope", p=lo-storage["total_discharge_e_mw"],
             q=(lo-storage["tes_discharge_e_mw"])*tan, background=1.0)])
    for bg in (1.0, .5, .1, 0.0):
        pts.append(dict(name=f"aidc_off_bg{bg:g}", p=0.0, q=0.0, background=bg))
    return pts


def check_candidate(cfg, candidate, points, *, current_factor=1.0, resistance_factor=1.0):
    rows = []
    for point in points:
        row, _ = snapshot(cfg, candidate, point["p"], point["q"], background=point["background"],
                          current_factor=current_factor, resistance_factor=resistance_factor)
        rows.append({"name": point["name"], "background": point["background"], **row})
    return dict(candidate=candidate, all_feasible=all(r["feasible"] for r in rows),
                unresolved=sum(not r.get("converged", False) or "failure" in r for r in rows),
                failures=[r["name"] for r in rows if not r["feasible"]], points=rows)


def select(records, key):
    passing = [r for r in records if r[key]["all_feasible"]]
    return min(passing, key=lambda r: rank(r["candidate"]))["candidate"] if passing else None


def replay(cfg, candidate, p, q, name, **kwargs):
    p, q = np.asarray(p), np.asarray(q)
    if p.ndim != 1 or p.shape != q.shape or len(p) == 0 or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Invalid replay arrays")
    net = build_network(cfg, candidate, **kwargs)
    idx = core.add_aidc(net, 2, float(p[0]), cfg["fixed"]["facility_lagging_pf"])
    rows = []
    for k, (pk,qk) in enumerate(zip(p,q)):
        net.load.at[idx,"p_mw"], net.load.at[idx,"q_mvar"] = float(pk), float(qk)
        rows.append(dict(interval=k, time_h=k/12, pcc_p_mw=float(pk), pcc_q_mvar=float(qk), **solve(net,cfg)))
        if k % 480 == 0:
            print(f"{name}: {k}/{len(p)}", flush=True)
    core.write_json(OUT / f"{name}.json", rows)
    summary = native.summarize(rows)
    good = [r for r in rows if r.get("converged") and "failure" not in r]
    summary.update(max_voltage_pu=max(r["max_voltage_pu"] for r in good) if good else None,
                   max_root_current_per_circuit_ka=max(r["root_current_per_circuit_ka"] for r in good) if good else None,
                   max_root_loading_percent=max(r["root_loading_percent"] for r in good) if good else None,
                   max_other_loading_percent=max(r["other_max_loading_percent"] for r in good) if good else None,
                   max_q_balance_error_mvar=max(abs(r["reactive_power_balance_error_mvar"]) for r in good) if good else None,
                   network_loss_energy_mwh=sum(r["network_losses_mw"] for r in good)/12,
                   file=f"{name}.json")
    return summary


def sensitivities(cfg, candidate, peak, storage):
    result = []
    pf0 = cfg["fixed"]["facility_lagging_pf"]
    def add(kind, value, pf=pf0, **kwargs):
        for mode in ("no_storage_peak", "peak_charge_envelope"):
            charge = mode == "peak_charge_envelope"
            p = peak + (storage["total_charge_e_mw"] if charge else 0)
            q = (peak + (storage["tes_charge_e_mw"] if charge else 0))*math.tan(math.acos(pf))
            row, _ = snapshot(cfg, candidate, p, q, **kwargs)
            result.append(dict(kind=kind,value=value,mode=mode,**row))
    s = cfg["sensitivity"]
    for bg,pf in itertools.product(s["background_scales"],s["facility_power_factors"]):
        add("background_pf",dict(background=bg,pf=pf),pf=pf,background=bg)
    for key, param in (("route_lengths_km","length"),("current_factors","current_factor"),
                       ("resistance_factors","resistance_factor"),("other_branch_limits_ka","other_current"),
                       ("head_limits_mva","head"),("source_voltage_offsets_pu","source_offset")):
        for value in s[key]:
            add(key,value,**{param:value})
    add("combined_stress",dict(background=1.1,pf=.95,length=2,current_factor=.9,resistance_factor=1.1),
        pf=.95,background=1.1,length=2,current_factor=.9,resistance_factor=1.1)
    # An outage is assessed as a limitation, never included in normal qualification.
    if candidate["area_mm2"] and candidate["circuits"] > 1:
        outage = dict(candidate,circuits=candidate["circuits"]-1)
        row,_ = snapshot(cfg,outage,peak,peak*math.tan(math.acos(pf0)))
        result.append(dict(kind="one_parallel_circuit_out_of_service",value=1,mode="no_storage_peak",**row))
    return result


def static_limit(cfg,candidate):
    tan = math.tan(math.acos(cfg["fixed"]["facility_lagging_pf"]))
    def point(p):
        return snapshot(cfg,candidate,p,p*tan)[0]
    coarse = [point(float(p)) for p in np.linspace(0,50,11)]
    flags = [r["feasible"] for r in coarse]
    if not flags[0] or any(flags[j] and not flags[j-1] for j in range(1,len(flags))):
        return dict(capacity_mw=None,coarse=coarse,reason="baseline_infeasible_or_nonmonotonic")
    if all(flags):
        return dict(capacity_mw=50,right_censored=True,coarse=coarse)
    first = flags.index(False)
    lo,hi = float((first-1)*5),float(first*5)
    while hi-lo > .001:
        mid = (lo+hi)/2
        if point(mid)["feasible"]: lo=mid
        else: hi=mid
    return dict(capacity_mw=lo,first_infeasible_mw=hi,last_feasible=point(lo),first_infeasible=point(hi),
                note="Single-period fixed-PF AC boundary, not DOE or multi-period storage feasibility.")


def main():
    cfg = read_config()
    OUT.mkdir(parents=True,exist_ok=True)
    protected = [ROOT/"configs/modified_ieee33_candidate_v1.json", ROOT/"configs/ieee33_1266kv_screen_v1.json",
                 ROOT/"configs/ess_tes_pcc_scheduling_v3_eta90.json",ROOT/"configs/pcc_tes_operating_candidate_v1.json"]
    for folder in ("outputs/modified_ieee33_candidate_v1","outputs/ieee33_1266kv_screen_v1","outputs/ess_tes_pcc_scheduling_v3_eta90"):
        protected.extend(p for p in (ROOT/folder).rglob("*") if p.is_file())
    before = {str(p.relative_to(ROOT)):core.digest(p) for p in protected}
    data,storage = inputs()
    p0 = data["no_storage_pcc_mw"]
    pf = cfg["fixed"]["facility_lagging_pf"]
    tan = math.tan(math.acos(pf))
    points = operating_points(cfg,p0,storage)
    core.write_json(OUT/"model_contract.json",cfg)
    core.write_json(OUT/"selection_operating_points.json",points)
    records = []
    for i,c in enumerate(candidates(cfg)):
        nom = check_candidate(cfg,c,points)
        alt = check_candidate(cfg,c,points,current_factor=cfg["search"]["alternative_current_factor"],
                              resistance_factor=cfg["search"]["alternative_resistance_factor"])
        records.append(dict(candidate=c,nominal=nom,allowance=alt))
        if i % 22 == 0: print(f"Candidates {i}/{len(candidates(cfg))}",flush=True)
    core.write_json(OUT/"candidate_screen.json",records)
    selected = select(records,"nominal")
    alternative = select(records,"allowance")
    if selected is None:
        core.write_json(OUT/"results.json",dict(status="no_feasible_candidate",search=cfg["search"]))
        raise RuntimeError("No passing candidate; do not silently expand the design space")
    chosen = {"nominal_minimum":selected}
    if alternative: chosen["allowance_alternative"] = alternative
    print("Selected: "+json.dumps(chosen),flush=True)
    prefix = "constant_25_5_joint"
    pj = data[prefix+"_pcc_mw"]
    qj = native.pcc_q(pj,data[prefix+"_ess_charge_mw"],data[prefix+"_ess_discharge_mw"],pf)
    streams = {"no_storage":(p0,p0*tan),"existing_joint":(pj,qj),
               "max_charge_envelope":(p0+storage["total_charge_e_mw"],(p0+storage["tes_charge_e_mw"])*tan)}
    evaluations = {}
    for label,c in chosen.items():
        nominal = next(r["nominal"] for r in records if r["candidate"] == c)
        entry = dict(candidate=c,selection_points=nominal, sensitivity=sensitivities(cfg,c,float(p0.max()),storage),
                     static_limit_not_doe=static_limit(cfg,c),replays={})
        for name,(p,q) in streams.items():
            entry["replays"][name] = replay(cfg,c,p,q,f"{label}_{name}")
        if label == "allowance_alternative":
            entry["replays"]["derated_charge_envelope"] = replay(cfg,c,*streams["max_charge_envelope"],
                   f"{label}_derated_charge_envelope",current_factor=.9,resistance_factor=1.1)
        row,net = snapshot(cfg,c,float(p0.max()),float(p0.max())*tan)
        pp.to_json(net,str(OUT/f"{label}_peak_network.json"))
        nr_v=net.res_bus.vm_pu.to_numpy().copy()
        nr_i=net.res_line.i_ka.to_numpy().copy()
        pp.runpp(net,algorithm="bfsw",init="flat",numba=False,max_iteration=200,tolerance_mva=1e-8)
        entry["solver_crosscheck"] = dict(max_voltage_difference_pu=float(np.max(np.abs(net.res_bus.vm_pu-nr_v))),
                                        max_line_current_difference_ka=float(np.max(np.abs(net.res_line.i_ka-nr_i))))
        entry["peak_voltage_profile_pu"]=nr_v.tolist()
        entry["root_parameters"]=net.line.iloc[0].to_dict()
        evaluations[label]=entry
    unchanged = {k:core.digest(ROOT/k)==v for k,v in before.items()}
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),status="candidate_screen_and_ac_validation_complete_pending_review",
                  config_sha256=core.digest(CONFIG),frequency_hz=60,nominal_voltage_kv=12.66,
                  candidates_count=len(records),candidate_operating_points_count=len(points),
                  nominal_passing_count=sum(r["nominal"]["all_feasible"] for r in records),
                  allowance_passing_count=sum(r["allowance"]["all_feasible"] for r in records),
                  frozen_profile=dict(mean_mw=float(p0.mean()),peak_mw=float(p0.max()),min_mw=float(p0.min()),intervals=len(p0),step_s=300),
                  storage=storage,evaluations=evaluations,selection_rule=cfg["search"]["ranking"],
                  protected_input_hashes=before,protected_inputs_unchanged=unchanged,
                  source_capacity_increased=False,source_transformer_explicitly_modeled=False,
                  aidc_rescaled=False,storage_resized=False,storage_reoptimized=False,doe_implemented=False,
                  full_charge_envelope_note="Electrical stress envelope only: every interval at maximum charge is not an executable SOC-constrained schedule.")
    core.write_json(OUT/"results.json",result)
    assert all(unchanged.values())
    for e in evaluations.values():
        assert e["solver_crosscheck"]["max_voltage_difference_pu"] < 1e-6
        for s in e["replays"].values():
            assert s["feasible_intervals"]==len(p0) and s["unresolved_intervals"]==0
    print(json.dumps(core.serial({k:{"candidate":v["candidate"],"replays":v["replays"]} for k,v in evaluations.items()}),indent=2),flush=True)


if __name__ == "__main__":
    main()
