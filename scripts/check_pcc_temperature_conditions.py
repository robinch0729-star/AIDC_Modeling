"""Read-only temperature-condition audit of the existing PCC/TES candidate.

Writes a separate numerical audit; it neither changes model parameters nor reruns
the full five-day simulation. Ideal temperature-window calculations are not
guarantees of deliverable constant power in a real or simulated facility.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pcc_temperature_conditions_v1"
CONFIG = ROOT / "configs/pcc_cooling_tes_candidate_v1.json"
LEGACY = ROOT / "outputs/pcc_cooling_tes_candidate_v1"
sys.path.insert(0, str(ROOT / "src"))
from aidc_facility.pcc_cooling_tes import (  # noqa: E402
    maximum_discharge_tank_temperature,
    tank_capacity,
    tank_step,
    validate,
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def protected_files() -> list[Path]:
    files = [CONFIG, ROOT / "src/aidc_facility/pcc_cooling_tes.py",
             ROOT / "tests/test_pcc_cooling_tes.py",
             ROOT / "outputs/it_5min_operating_baseline_v1/baseline_1s.npz"]
    for dirname in ("pcc_cooling_tes_candidate_v1", "pcc_capacity_cop_review_v1",
                    "it_pcc_boundary_audit_v1"):
        directory = ROOT / "outputs" / dirname
        if directory.exists():
            files.extend(p for p in directory.rglob("*") if p.is_file())
    return sorted(set(files))


def snapshots(files: list[Path]) -> dict[str, dict]:
    return {p.relative_to(ROOT).as_posix(): {
        "bytes": p.stat().st_size, "sha256": digest(p)} for p in files}


def bounds(values: np.ndarray) -> dict:
    return {"minimum": float(values.min()), "maximum": float(values.max())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protected = protected_files()
    before = snapshots(protected)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate(config)
    t, r = config["tes"], config["room"]
    cap = tank_capacity(t)
    alpha = t["hx_effectiveness"] * t["maximum_flow_kg_s"] * t["water_cp_j_kg_k"] / 1e6
    warm, initial = t["reference_warm_c"], t["initial_c"]
    reserve = t["minimum_reserve_mwh_th"]
    reserve_tmax = warm - reserve / cap
    supply_tmax = maximum_discharge_tank_temperature(t)
    inventory = cap * (warm - initial)

    charging = []
    for temperature in (7.0, 7.5, 8.5, 9.5):
        raw = alpha * (temperature - t["source_chilled_c"])
        charging.append({
            "tank_c": temperature,
            "hx_flow_ceiling_mw_th": raw,
            "hx_flow_and_nameplate_ceiling_mw_th": min(raw, t["maximum_charge_mw_th"]),
            "not_a_dispatch_limit": "End temperature, minimum tank temperature, chiller headroom and charging control still apply.",
        })

    discharging = []
    for temperature in (7.5, 9.0, t["load_return_c"] - 5.0 / alpha, supply_tmax):
        raw = alpha * (t["load_return_c"] - temperature)
        supply = t["load_return_c"] - t["hx_effectiveness"] * (t["load_return_c"] - temperature)
        discharging.append({
            "tank_c": temperature,
            "hx_flow_ceiling_mw_th": raw,
            "hx_flow_and_nameplate_ceiling_mw_th": min(raw, t["maximum_discharge_mw_th"]),
            "load_side_outlet_c_at_configured_effectiveness": supply,
            "within_instantaneous_supply_limit": supply <= t["maximum_discharge_supply_c"] + 1e-10,
        })

    power_cutoffs = []
    constant_discharge = []
    for q in (3.0, 5.0):
        t_charge = t["source_chilled_c"] + q / alpha
        t_discharge = t["load_return_c"] - q / alpha
        end = min(t_discharge, supply_tmax, reserve_tmax)
        usable = max(0.0, cap * (end - initial))
        power_cutoffs.append({
            "thermal_power_mw_th": q,
            "minimum_tank_c_for_charging_from_hx_flow_only": t_charge,
            "maximum_tank_c_for_discharging_from_hx_flow_only": t_discharge,
        })
        constant_discharge.append({
            "thermal_power_mw_th": q,
            "limiting_end_tank_c": end,
            "limiting_condition": "supply_temperature" if supply_tmax <= min(t_discharge, reserve_tmax) else "hx_flow",
            "ideal_energy_from_initial_mwh_th": usable,
            "ideal_duration_h_ignoring_heat_leak_and_other_constraints": usable / q,
            "not_guaranteed_dispatch_duration": True,
        })

    supply_sensitivity = []
    for supply in (8.0, 9.0, 10.0):
        ceiling = t["load_return_c"] - (t["load_return_c"] - supply) / t["hx_effectiveness"]
        end_with_reserve = min(ceiling, reserve_tmax)
        usable = max(0.0, cap * (ceiling - initial))
        usable_reserve = max(0.0, cap * (end_with_reserve - initial))
        supply_sensitivity.append({
            "maximum_discharge_supply_c": supply,
            "maximum_tank_c_from_supply": ceiling,
            "ideal_temperature_window_energy_mwh_th": usable,
            "ideal_energy_with_configured_reserve_mwh_th": usable_reserve,
            "reserve_reduces_energy_mwh_th": usable - usable_reserve,
            "remaining_inventory_at_supply_cutoff_mwh_th": cap * (warm - ceiling),
            "scope": "Independent algebraic sensitivity; no configuration or legacy profile changed.",
        })

    columns = json.loads((LEGACY / "internal_columns.json").read_text(encoding="utf-8"))
    with np.load(LEGACY / "internal_profiles.npz", allow_pickle=False) as bundle:
        existing = bundle["with_tes"]
    series = {name: existing[:, i] for i, name in enumerate(columns)}
    charging_mask = series["tes_charge_mw_th"] > 1e-6
    discharging_mask = series["tes_discharge_mw_th"] > 1e-6
    profile_audit = {
        "scope": "Audit of previously generated with_tes arrays, not a newly simulated experiment.",
        "source_npz_key": "with_tes",
        "internal_step_s": config["internal_step_s"],
        "sample_count": len(existing),
        "duration_h": len(existing) * config["internal_step_s"] / 3600,
        "tank_temperature_c_including_start_and_end": bounds(np.r_[series["tank_start_c"], series["tank_end_c"]]),
        "room_temperature_c_including_start_and_end": bounds(np.r_[series["room_start_c"], series["room_end_c"]]),
        "stored_inventory_mwh_th_including_start_and_end": bounds(np.r_[series["tes_energy_start_mwh_th"], series["tes_energy_end_mwh_th"]]),
        "discharge_supply_c_active_discharge_only": bounds(series["tes_discharge_supply_c"][discharging_mask]),
        "tes_branch_flow_kg_s_active_charge_only": bounds(series["tes_flow_kg_s"][charging_mask]),
        "tes_branch_flow_kg_s_active_discharge_only": bounds(series["tes_flow_kg_s"][discharging_mask]),
        "active_charge_samples": int(charging_mask.sum()),
        "active_discharge_samples": int(discharging_mask.sum()),
        "maximum_actual_charge_mw_th": float(series["tes_charge_mw_th"].max()),
        "maximum_actual_discharge_mw_th": float(series["tes_discharge_mw_th"].max()),
        "room_temperature_violations": int(((series["room_end_c"] < r["min_c"] - 1e-8) | (series["room_end_c"] > r["max_c"] + 1e-8)).sum()),
        "discharge_supply_violations": int((series["tes_discharge_supply_c"][discharging_mask] > t["maximum_discharge_supply_c"] + 1e-7).sum()),
        "tes_branch_flow_violations": int((series["tes_flow_kg_s"] > t["maximum_flow_kg_s"] + 1e-6).sum()),
        "final_tank_c": float(series["tank_end_c"][-1]),
    }

    independent_assertions = []
    def check(name: str, condition: bool) -> None:
        assert condition, name
        independent_assertions.append({"name": name, "passed": True})
    check("tank_heat_capacity_units", abs(cap - 3.4833333333333334) < 1e-12)
    check("hx_flow_coefficient_units", abs(alpha - 1.9855) < 1e-12)
    check("initial_inventory", abs(inventory - 15.675) < 1e-12)
    check("finite_hx_supply_tank_ceiling", abs(supply_tmax - 9.894736842105264) < 1e-12)
    check("reserve_not_binding_before_supply_cutoff", reserve_tmax > supply_tmax)
    check("five_mw_discharge_flow_limit_before_supply_limit", constant_discharge[1]["limiting_end_tank_c"] < supply_tmax)
    check("supply_8_9_10_energy_monotonic", all(supply_sensitivity[i]["ideal_temperature_window_energy_mwh_th"] < supply_sensitivity[i+1]["ideal_temperature_window_energy_mwh_th"] for i in range(2)))
    check("saved_profile_is_finite", bool(np.isfinite(existing).all()))
    check("saved_profile_columns_match", existing.shape[1] == len(columns))
    check("saved_profile_tank_inventory_consistent", bool(np.allclose(series["tes_energy_end_mwh_th"], cap * (warm - series["tank_end_c"]), rtol=0, atol=1e-10)))
    check("saved_profile_discharge_supply_formula", bool(np.allclose(series["tes_discharge_supply_c"][discharging_mask], t["load_return_c"] - t["hx_effectiveness"] * (t["load_return_c"] - np.maximum(series["tank_start_c"][discharging_mask], series["tank_end_c"][discharging_mask])), rtol=0, atol=1e-10)))
    # Small direct function checks are not a full-profile rerun.
    charging_at_target = tank_step(initial, -3.0, config["internal_step_s"] / 3600, t, 18.0, 26.0)
    check("three_mw_charge_at_target_is_flow_limited", abs(charging_at_target["net"]) < 3.0)
    blocked = tank_step(supply_tmax + 0.01, 3.0, config["internal_step_s"] / 3600, t, 18.0, 26.0)
    check("tank_above_supply_cutoff_has_no_discharge", blocked["net"] < 1e-7)

    test_run = subprocess.run([sys.executable, "-B", str(ROOT / "tests/test_pcc_cooling_tes.py")], cwd=ROOT, text=True, capture_output=True, check=False)
    check("existing_pcc_tests_exit_zero", test_run.returncode == 0)
    after = snapshots(protected)
    check("all_original_files_preserved", before == after)

    notes = [
        "All room, source-water, return-water and TES temperature values remain scenario assumptions; this audit selects no new parameter.",
        "The modeled room temperature is one equivalent room state, not verified equipment inlet temperature or a hot-spot temperature.",
        "The existing coil heat-removal calculation is algebraic and has no explicit chilled-water inlet/outlet temperature or main-loop mass-flow model.",
        "The configured 500 kg/s maximum flow is a TES-branch/HX limit; it is not the main chilled-water-loop flow.",
        "The 12 C reference warm temperature defines an energy zero; its equality to load return temperature is a current scenario choice.",
        "A TES heat-exchanger outlet supply limit of 10 C does not independently prove CRAH cooling adequacy or server inlet-temperature compliance.",
        "Instantaneous HX power limits omit finite-step end-temperature effects, ambient heat leak, minimum tank temperature, chiller headroom and dispatch control.",
        "Ideal constant-output duration omits heat leak and other facility constraints and is not a guaranteed capability.",
        "At tank minimum temperature 7 C, nonzero sustained charging would need an incoming heat load; the instantaneous HX ceiling is not permission to cool below 7 C.",
        "Source code, configuration and legacy results have not been changed; full-period outputs are an existing-results audit only.",
    ]
    payload = {
        "audit_version": "pcc_temperature_conditions_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": CONFIG.relative_to(ROOT).as_posix(),
        "new_full_period_simulation": False,
        "model_parameters_changed": False,
        "units": {"thermal_power": "MW_th", "thermal_energy": "MWh_th", "temperature": "degC", "mass_flow": "kg/s"},
        "tank_heat_capacity_mwh_per_k": cap,
        "hx_max_flow_coefficient_mw_per_k": alpha,
        "initial_reference_inventory_mwh_th": inventory,
        "maximum_tank_c_from_discharge_supply": supply_tmax,
        "maximum_tank_c_from_reserve": reserve_tmax,
        "inventory_at_supply_cutoff_mwh_th": cap * (warm - supply_tmax),
        "ideal_usable_temperature_window_mwh_th": cap * (supply_tmax - initial),
        "charging_hx_limits": charging,
        "discharging_hx_limits": discharging,
        "target_power_temperature_cutoffs": power_cutoffs,
        "ideal_constant_discharge_windows": constant_discharge,
        "maximum_supply_temperature_sensitivity": supply_sensitivity,
        "existing_enabled_tes_profile_audit": profile_audit,
        "small_function_checks": {"charge_3mw_from_7p5c_one_step": charging_at_target, "discharge_above_supply_temperature_cutoff_one_step": blocked},
        "model_interpretation_and_limits": notes,
    }
    verification = {
        "independent_assertions": independent_assertions,
        "independent_assertion_count": len(independent_assertions),
        "existing_test_command": [sys.executable, "-B", "tests/test_pcc_cooling_tes.py"],
        "existing_test_exit_code": test_run.returncode,
        "existing_test_stdout": test_run.stdout,
        "existing_test_stderr": test_run.stderr,
        "original_file_count": len(protected),
        "original_files_unchanged": before == after,
        "full_period_model_rerun": False,
        "physical_facility_validation": False,
    }
    manifest = {"originals_before": before, "originals_after": after,
                "new_script": {"path": Path(__file__).relative_to(ROOT).as_posix(), "sha256": digest(Path(__file__))},
                "original_files_unchanged": before == after}
    for filename, value in (("numeric_checks.json", payload), ("verification.json", verification), ("source_manifest.json", manifest)):
        (OUT / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_directory": str(OUT), "original_files_preserved": len(protected), "assertions_passed": len(independent_assertions), "existing_tests_exit_code": test_run.returncode, "constant_discharge_windows": constant_discharge, "existing_profile": profile_audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
