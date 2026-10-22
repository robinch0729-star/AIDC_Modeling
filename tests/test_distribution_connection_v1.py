"""Checks for frozen benchmark boundaries and transparent scenario selection."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandapower as pp
import pandas as pd
import pytest
import simbench as sb

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/grid_connection_screen_v1"


@pytest.fixture(scope="module")
def data():
    cfg=json.loads((ROOT/"configs/distribution_connection_v1.json").read_text(encoding="utf-8"))
    sel=json.loads((OUT/"selection.json").read_text(encoding="utf-8"))
    screen=json.loads((OUT/"screening_results.json").read_text(encoding="utf-8"))
    net=pp.from_json(str(OUT/"selected_network_mean.json"))
    original=sb.get_simbench_net(cfg["benchmark"]["code"])
    return cfg,sel,screen,net,original


def test_original_topology_loads_and_equipment_preserved(data):
    _,_,_,net,original=data
    for name in ("line","bus","load","sgen","switch","ext_grid"):
        a=net[name].loc[original[name].index, original[name].columns]
        pd.testing.assert_frame_equal(a,original[name],check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-10)
    cols=[c for c in original.trafo.columns if c not in ("tap_pos","tap_changer_type")]
    pd.testing.assert_frame_equal(net.trafo[cols],original.trafo[cols],check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-10)
    assert len(net.line)==len(original.line)+1
    assert len(net.bus)==len(original.bus)+1
    assert len(net.load)==len(original.load)+1
    assert len(net.trafo)==len(original.trafo)


def test_tap_selection_not_feasibility_or_storage_tuned(data):
    cfg,sel,_,net,_=data
    rows=sel["calibration_rows"]
    assert sorted(r["study_synchronized_ratio_tap_setting"] for r in rows)==list(range(-9,10))
    target=sel["source_lv_setpoint_pu"]
    best=min((r for r in rows if r["converged"]),key=lambda r:abs(r["main_bus_vm_pu"]-target))
    assert best["study_synchronized_ratio_tap_setting"]==sel["selected_synchronized_ratio_tap"]
    assert list(net.trafo.tap_pos)==[-2,-2]
    assert list(net.trafo.tap_changer_type)==["Ratio","Ratio"]
    assert cfg["tap_operating_assumption"]["selected_synchronized_tap"]==-2


def test_frozen_aidc_input_and_loss_boundary(data):
    cfg,_,screen,net,original=data
    pcc_load=net.load.index.difference(original.load.index)[0]
    p=net.load.at[pcc_load,"p_mw"]
    assert p==pytest.approx(cfg["aidc_interface"]["pcc_mean_no_storage_mw"])
    assert net.load.at[pcc_load,"q_mvar"]==pytest.approx(p*math.tan(math.acos(.98)))
    assert not cfg["connection"]["site_transformer_added_to_network"]
    assert cfg["aidc_interface"]["ess_q_mvar"]==0
    assert not cfg["aidc_interface"]["profile_rescaling"]
    for name,sha in screen["source_files"].items():
        if not name.startswith("scripts/"):
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha


def test_selected_mean_and_peak_reproduce(data):
    cfg,sel,_,net,original=data
    idx=net.load.index.difference(original.load.index)[0]
    for demand,p in (("selected_mean",cfg["aidc_interface"]["pcc_mean_no_storage_mw"]),("selected_peak",cfg["aidc_interface"]["pcc_peak_5min_no_storage_mw"])):
        n=copy.deepcopy(net)
        n.load.at[idx,"p_mw"]=p
        n.load.at[idx,"q_mvar"]=p*math.tan(math.acos(.98))
        pp.runpp(n,numba=False)
        r=next(r for r in sel["selected_case_checks"] if r["background_case"]=="native" and r["demand"]==demand)
        mv=n.bus.index[n.bus.vn_kv.between(10,30)]
        assert n.converged
        assert n.res_bus.loc[mv,"vm_pu"].min()==pytest.approx(r["native_mv_voltage_min_pu"],abs=1e-8)
        assert n.res_bus.loc[mv,"vm_pu"].between(.965,1.055).all()
        assert n.res_line.loading_percent.max()<100
        assert n.res_trafo.loading_percent.max()<100


def test_known_failures_are_not_hidden(data):
    _,sel,screen,_,_=data
    for case in ("lW","lPV"):
        r=next(r for r in sel["selected_case_checks"] if r["background_case"]==case and r["demand"]=="selected_peak")
        assert r["native_mv_voltage_max_pu"]>1.055
        assert not r["all_native_mv_constraints_satisfied"]
    raw=next(r for r in screen["cases"] if r["network"]==sel["network_code"] and r["background_case"]=="native" and r["demand"]=="peak" and r["main_bus"]==2)
    assert raw["native_mv_voltage_min_pu"]<.965
    assert raw["pcc_vm_pu"]>.965


def test_three_circuits_and_n_minus_one_not_claimed(data):
    cfg,_,_,net,original=data
    row=net.line.loc[net.line.index.difference(original.line.index)[0]]
    assert row.parallel==3
    assert row.length_km==1
    assert row.max_i_ka==.362
    assert not cfg["connection"]["n_minus_one_claimed"]
    i=cfg["aidc_interface"]["pcc_peak_5min_no_storage_mw"]/(math.sqrt(3)*20*.98)
    assert 2*.362<i<3*.362
