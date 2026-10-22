"""Read-only-to-existing-models AC screening of candidate AIDC grid connections.

This is neither OPF nor DOE or time-series validation. Native SimBench topology,
ratings and background parameters are retained. Only a declared dedicated AIDC
connection is added. The original profile-derived AIDC PCC power is not scaled.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandapower as pp
import simbench as sb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/grid_connection_screen_v1"
INPUT = ROOT / "outputs/ess_tes_pcc_scheduling_v3_eta90/summary.json"
LINE_TYPE = "NA2XS2Y 1x185 RM/25 12/20 kV"
CODES = ("1-MV-comm--0-sw", "1-MV-semiurb--0-sw")
TOL = 1e-7


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def records(frame, columns):
    # pandas' JSON writer consistently handles NA and numpy scalar types.
    return json.loads(frame.reset_index().to_json(orient="records")) if columns is None else json.loads(
        frame.loc[:, [c for c in columns if c in frame]].reset_index().to_json(orient="records")
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_digest(frame):
    return hashlib.sha256(frame.to_json(orient="split", double_precision=15).encode()).hexdigest()


def metadata(net, code):
    mv = net.bus.vn_kv.between(10, 30)
    return {
        "code": code,
        "counts": {k: len(net[k]) for k in ("bus", "line", "trafo", "load", "sgen", "switch")},
        "native_table_sha256": {k: frame_digest(net[k]) for k in ("bus", "line", "trafo", "load", "sgen", "switch", "ext_grid", "loadcases")},
        "native_bus_records": records(net.bus, ["name", "vn_kv", "min_vm_pu", "max_vm_pu", "in_service"]),
        "transformers": records(net.trafo, ["name", "hv_bus", "lv_bus", "sn_mva", "vn_hv_kv", "vn_lv_kv", "in_service", "tap_pos", "max_loading_percent"]),
        "bus_couplers": records(net.switch[net.switch.et == "b"], ["name", "bus", "element", "closed", "et"]),
        "ext_grid": records(net.ext_grid, ["name", "bus", "vm_pu", "va_degree", "in_service"]),
        "native_mv_voltage_bounds": {"min_values": sorted(net.bus.loc[mv, "min_vm_pu"].unique().tolist()), "max_values": sorted(net.bus.loc[mv, "max_vm_pu"].unique().tolist())},
        "line_type": dict(net.std_types["line"][LINE_TYPE]),
        "loadcase_factors": records(net.loadcases, None),
        "native_scaling_values": {k: sorted(net[k].scaling.unique().tolist()) for k in ("load", "sgen")},
        "background_native_totals": {f"{k}_{c}": float((net[k][c] * net[k].scaling).sum()) for k in ("load", "sgen") for c in ("p_mw", "q_mvar")},
    }


def add_aidc(net, main_bus, p_mw, pf=0.98, length_km=1.0, derating=1.0):
    main = net.bus.loc[main_bus]
    pcc = pp.create_bus(net, vn_kv=float(main.vn_kv), name="AIDC PCC (study assumption)",
                        min_vm_pu=float(main.min_vm_pu), max_vm_pu=float(main.max_vm_pu))
    branch = pp.create_line(net, from_bus=main_bus, to_bus=pcc, length_km=length_km,
                            std_type=LINE_TYPE, parallel=3, df=derating,
                            name="AIDC dedicated three-parallel-circuit connection (study assumption)",
                            max_loading_percent=100.0)
    q_mvar = p_mw * math.tan(math.acos(pf))
    load = pp.create_load(net, bus=pcc, p_mw=p_mw, q_mvar=q_mvar,
                          name="AIDC frozen PCC demand, storage idle", const_z_p_percent=0,
                          const_i_p_percent=0, const_z_q_percent=0, const_i_q_percent=0)
    return pcc, branch, load


def snapshot(raw, absolute, case):
    net = copy.deepcopy(raw)
    if case != "native":
        for (table, column), values in absolute.items():
            net[table].loc[values.columns, column] = values.loc[case].values
    return net


def solve(net, code, case, demand_label, p_mw, main_bus=None, pf=0.98,
          length_km=1.0, derating=1.0, tap_setting=None):
    if tap_setting is not None:
        # SimBench 1.6.3 imports tap_changer_type=None into pandapower 3.5.6.
        # That leaves tap_pos ineffective. Ratio activation is explicit here;
        # original step, side, limits and winding/impedance ratings are retained.
        net.trafo["tap_changer_type"] = "Ratio"
        net.trafo["tap_pos"] = tap_setting
    meta = {
        "network": code, "background_case": case, "demand": demand_label,
        "aidc_p_mw": p_mw, "facility_lagging_pf": pf if p_mw else None,
        "aidc_q_mvar": p_mw * math.tan(math.acos(pf)),
        "main_bus": main_bus, "main_bus_name": str(net.bus.at[main_bus, "name"]) if main_bus is not None else None,
        "connection_length_km": length_km if main_bus is not None else None,
        "connection_parallel": 3 if main_bus is not None else None,
        "connection_derating": derating if main_bus is not None else None,
        "background_p_load_mw": float((net.load.p_mw * net.load.scaling).sum()),
        "background_q_load_mvar": float((net.load.q_mvar * net.load.scaling).sum()),
        "background_p_generation_mw": float((net.sgen.p_mw * net.sgen.scaling).sum()),
        "slack_vm_pu": net.ext_grid.vm_pu.tolist(),
        "fixed_native_taps": net.trafo.tap_pos.tolist(),
        "tap_changer_types": net.trafo.tap_changer_type.tolist(),
        "study_synchronized_ratio_tap_setting": tap_setting,
    }
    pcc = branch = None
    if main_bus is not None:
        pcc, branch, _ = add_aidc(net, main_bus, p_mw, pf, length_km, derating)
    try:
        pp.runpp(net, algorithm="nr", calculate_voltage_angles=True, init="auto",
                 numba=False, max_iteration=50, tolerance_mva=1e-8,
                 check_connectivity=True, trafo_loading="current")
    except Exception as exc:
        return {**meta, "converged": False, "error": repr(exc)}, net
    mv = net.bus.index[net.bus.vn_kv.between(10, 30) & net.bus.in_service]
    vm = net.res_bus.loc[mv, "vm_pu"]
    low = mv[vm < net.bus.loc[mv, "min_vm_pu"] - TOL]
    high = mv[vm > net.bus.loc[mv, "max_vm_pu"] + TOL]
    unsupplied = mv[vm.isna()]
    overloaded_lines = net.line.index[net.res_line.loading_percent > net.line.max_loading_percent.fillna(100) + TOL]
    overloaded_trafos = net.trafo.index[net.res_trafo.loading_percent > net.trafo.max_loading_percent.fillna(100) + TOL]
    def voltage_rows(indices):
        return [{"bus": int(i), "name": str(net.bus.at[i, "name"]), "vm_pu": float(vm.loc[i]),
                 "min_vm_pu": float(net.bus.at[i, "min_vm_pu"]), "max_vm_pu": float(net.bus.at[i, "max_vm_pu"])} for i in indices]
    result = {
        **meta, "converged": bool(net.converged),
        "native_mv_voltage_min_pu": float(vm.min()), "native_mv_voltage_max_pu": float(vm.max()),
        "min_voltage_bus": int(vm.idxmin()), "min_voltage_bus_name": str(net.bus.at[vm.idxmin(), "name"]),
        "under_voltage": voltage_rows(low), "over_voltage": voltage_rows(high),
        "unsupplied_mv_buses": unsupplied.tolist(),
        "max_line_loading_percent": float(net.res_line.loading_percent.max()),
        "max_line_index": int(net.res_line.loading_percent.idxmax()),
        "overloaded_lines": records(net.line.loc[overloaded_lines].join(net.res_line.loading_percent.rename("result_loading_percent")), ["name", "from_bus", "to_bus", "result_loading_percent", "max_loading_percent"]),
        "max_trafo_loading_percent": float(net.res_trafo.loading_percent.max()),
        "overloaded_trafos": overloaded_trafos.tolist(),
        "trafo_results": records(net.trafo[["name", "sn_mva"]].join(net.res_trafo[["p_hv_mw", "q_hv_mvar", "loading_percent"]]), None),
        "ext_grid_p_mw": float(net.res_ext_grid.p_mw.sum()),
        "ext_grid_q_mvar": float(net.res_ext_grid.q_mvar.sum()),
        "all_native_mv_constraints_satisfied": len(low) + len(high) + len(unsupplied) + len(overloaded_lines) + len(overloaded_trafos) == 0,
    }
    if pcc is not None:
        result.update(pcc_bus=int(pcc), pcc_vm_pu=float(net.res_bus.at[pcc, "vm_pu"]),
                      main_bus_vm_pu=float(net.res_bus.at[main_bus, "vm_pu"]),
                      connection_line=int(branch), connection_loading_percent=float(net.res_line.at[branch, "loading_percent"]),
                      connection_losses_mw=float(net.res_line.at[branch, "pl_mw"]),
                      connection_losses_mvar=float(net.res_line.at[branch, "ql_mvar"]))
    return result, net


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(INPUT.read_text(encoding="utf-8"))
    powers = {"mean": source["no_storage"]["mean_pcc_mw"], "peak": source["no_storage"]["peak_pcc_mw"]}
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": "AC snapshot screening, not OPF/DOE, equipment design, connection approval, N-1 or full time-series verification",
        "versions": {k: importlib.metadata.version(k) for k in ("pandapower", "simbench", "numpy", "scipy", "pandas")},
        "source_files": {str(INPUT.relative_to(ROOT)): digest(INPUT), str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
        "pcc_demand_mw": powers,
        "scenario_conventions": {
            "native": "Original SimBench load and generation values, native scaling, slack voltage and tap positions; parameter snapshot, not claimed coincident empirical measurement",
            "study_cases": "Official SimBench factors applied through get_absolute_values(profiles_instead_of_study_cases=False); no n1 case run because no contingency is simulated",
            "voltage": "Use native per-bus bounds, not general .95-1.05 pu or grid-code assertion",
            "connection": "Assumed new 1 km, three parallel native 185 mm2 cables, original feeder and transformer ratings untouched",
            "reactive_power": "Q_facility=P_facility*tan(acos(PF)); ESS has Q=0 and no ESS dispatch in this screening. Do not later apply PF to net power after ESS",
            "thermal": "Native steady-state current loading limit; df=1 is library rating, not a validated site installation ampacity",
            "controller": "Native fixed tap positions, no automatic OLTC control added",
            "tap_sensitivity": "Explicit static synchronized Ratio tap activation on both original transformers using their original 1.5% HV tap step and -9..9 range. Imported None type otherwise leaves tap_pos ineffective in pandapower3.5.6. Static operating assumption, not closed-loop controller validation.",
            "calendar": "Static official background cases not aligned to Azure workload dates",
        },
        "networks": {}, "cases": [],
    }
    for code in CODES:
        raw = sb.get_simbench_net(code)
        result["networks"][code] = metadata(raw, code)
        absolute = sb.get_absolute_values(raw, profiles_instead_of_study_cases=False)
        raw.pop("profiles", None)
        lv_buses = sorted(set(raw.trafo.loc[raw.trafo.in_service, "lv_bus"].astype(int)))
        cases = ["native", "hL", "hW", "hPV", "lW", "lPV"]
        for case in cases:
            r, _ = solve(snapshot(raw, absolute, case), code, case, "no_aidc", 0)
            result["cases"].append(r)
            for bus in lv_buses:
                for label, p_mw in powers.items():
                    r, solved = solve(snapshot(raw, absolute, case), code, case, label, p_mw, bus)
                    result["cases"].append(r)
                    if code == CODES[0] and case == "native" and bus == lv_buses[0] and label == "mean":
                        pp.to_json(solved, str(OUT / "commercial_candidate_mean_network.json"))
            for pf in (0.95, 1.0):
                r, _ = solve(snapshot(raw, absolute, case), code, case, "peak_pf_sensitivity", powers["peak"], lv_buses[0], pf=pf)
                result["cases"].append(r)
        if code == CODES[0]:
            for case in ("native", "hL"):
                for length, df in ((3.0, 1.0), (1.0, 0.8)):
                    r, _ = solve(snapshot(raw, absolute, case), code, case, "peak_connection_sensitivity", powers["peak"], lv_buses[0], length_km=length, derating=df)
                    result["cases"].append(r)
            for case in cases:
                for tap in (0, -1, -2):
                    for label, p_mw in {"no_aidc": 0.0, **powers}.items():
                        r, solved = solve(snapshot(raw, absolute, case), code, case,
                                          "ratio_tap_" + label, p_mw, lv_buses[0], tap_setting=tap)
                        result["cases"].append(r)
                        if case == "native" and tap == -1 and label == "mean":
                            pp.to_json(solved, str(OUT / "commercial_candidate_ratio_tap_minus1_mean_network.json"))
                        if case == "hL" and tap == -1 and label == "mean":
                            pp.to_json(solved, str(OUT / "commercial_hL_ratio_tap_minus1_mean_network.json"))
            for case in ("native", "hL"):
                for pf in (0.95, 1.0):
                    r, _ = solve(snapshot(raw, absolute, case), code, case,
                                 "ratio_tap_peak_pf_sensitivity", powers["peak"], lv_buses[0],
                                 pf=pf, tap_setting=-1)
                    result["cases"].append(r)
                for length, df in ((3.0, 1.0), (1.0, 0.8)):
                    r, _ = solve(snapshot(raw, absolute, case), code, case,
                                 "ratio_tap_peak_connection_sensitivity", powers["peak"], lv_buses[0],
                                 length_km=length, derating=df, tap_setting=-1)
                    result["cases"].append(r)
            # Calibrate the static transformer tap against the library's own
            # source-LV voltage target at native background + mean AIDC demand.
            # No network-feasibility or storage-benefit metric enters selection.
            target_vm = float(raw.trafo.autoTapSetp.iloc[0])
            min_tap = int(raw.trafo.tap_min.max())
            max_tap = int(raw.trafo.tap_max.min())
            calibration = []
            for tap in range(min_tap, max_tap + 1):
                r, _ = solve(snapshot(raw, absolute, "native"), code, "native",
                             "tap_calibration_mean", powers["mean"], lv_buses[0], tap_setting=tap)
                r["target_source_lv_vm_pu"] = target_vm
                r["absolute_source_lv_setpoint_error_pu"] = abs(r["main_bus_vm_pu"] - target_vm) if r["converged"] else None
                calibration.append(r)
            selected = min((r for r in calibration if r["converged"]),
                           key=lambda r: (r["absolute_source_lv_setpoint_error_pu"], abs(r["study_synchronized_ratio_tap_setting"])))
            selected_tap = selected["study_synchronized_ratio_tap_setting"]
            selected_cases = []
            for case in cases:
                for label, p_mw in {"no_aidc": 0.0, **powers}.items():
                    r, solved = solve(snapshot(raw, absolute, case), code, case,
                                      "selected_" + label, p_mw, lv_buses[0], tap_setting=selected_tap)
                    selected_cases.append(r)
                    if case == "native" and label == "mean":
                        pp.to_json(solved, str(OUT / "selected_network_mean.json"))
            unchanged_type_net = snapshot(raw, absolute, "native")
            unchanged_type_net.trafo["tap_pos"] = -1
            compatibility, _ = solve(unchanged_type_net, code, "native", "none_type_minus1_peak",
                                     powers["peak"], lv_buses[0])
            selection = {
                "network_code": code, "main_bus": lv_buses[0], "main_bus_name": str(raw.bus.at[lv_buses[0], "name"]),
                "calibration_background": "native", "calibration_demand_mw": powers["mean"],
                "selection_criterion": "Minimize absolute source-LV voltage error to native autoTapSetp at mean AIDC demand; no constraint-feasibility or ESS-benefit optimization",
                "source_lv_setpoint_pu": target_vm, "source_lv_setpoint_origin": "SimBench trafo.autoTapSetp",
                "scanned_native_tap_range": [min_tap, max_tap], "selected_synchronized_ratio_tap": selected_tap,
                "selected_source_lv_vm_pu": selected["main_bus_vm_pu"] ,
                "selected_absolute_setpoint_error_pu": selected["absolute_source_lv_setpoint_error_pu"],
                "tap_changer_compatibility": {"explanation": "SimBench imported tap_changer_type=None leaves tap_pos ineffective in pandapower3.5.6. Explicit Ratio activation retains native tap_side/step/range/ratings; fixed tap, not OLTC controller.", "none_type_minus1_peak_result": compatibility},
                "calibration_rows": calibration, "selected_case_checks": selected_cases,
                "selected_network": "selected_network_mean.json",
                "limitations": "Snapshot checks only. Neutral-tap network and all violations preserved. Static target-calibrated tap is not claimed feasible for all operating backgrounds or contingencies. No DSO OPF/DOE or ESS dispatch.",
            }
            (OUT / "selection.json").write_text(json.dumps(serial(selection), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            result["selection"] = {k: v for k, v in selection.items() if k not in ("calibration_rows", "selected_case_checks", "tap_changer_compatibility")}
        print(code, "screening complete", flush=True)
    result["exports"] = {
        "candidate_network": "commercial_candidate_mean_network.json",
        "ratio_tap_candidate_network": "commercial_candidate_ratio_tap_minus1_mean_network.json",
        "hL_ratio_tap_candidate_network": "commercial_hL_ratio_tap_minus1_mean_network.json",
        "export_status": "Candidate not certification of constraint feasibility; inspect case results. Profiles omitted to keep file compact."
    }
    target = OUT / "screening_results.json"
    target.write_text(json.dumps(serial(result), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    compact = [{k: r.get(k) for k in ("network", "background_case", "demand", "main_bus", "facility_lagging_pf", "connection_length_km", "connection_derating", "study_synchronized_ratio_tap_setting", "converged", "native_mv_voltage_min_pu", "native_mv_voltage_max_pu", "max_line_loading_percent", "max_trafo_loading_percent", "pcc_vm_pu", "main_bus_vm_pu", "all_native_mv_constraints_satisfied")} for r in result["cases"]]
    (OUT / "compact_results.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"cases": len(result["cases"]), "all_converged": all(r["converged"] for r in result["cases"]), "output": str(target)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
