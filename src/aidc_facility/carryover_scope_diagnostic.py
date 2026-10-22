"""Exploratory checks around frozen S1. No fit, power conversion or final sizing."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .nlr_processing_diagnostic import diagnose_window, time_average
from .carryover_replay import StateReplay, regular_arrivals

BANDS=("below_2", "2_to_30", "over_30_below_60", "60_to_100", "over_100")


def replay_diagnostic(obs,rate,parameters,*,power=1.,phase=.5,arrival_factor=1.):
    """Read virtual progress of unchanged engine; never change its work law."""
    r0,q0=int(obs.running_requests.iloc[0]),int(obs.waiting_requests.iloc[0])
    times=obs.replay_s.to_numpy(float)
    initial=((np.arange(r0)+.5)/r0)**power if r0 else np.array([])
    engine=StateReplay(regular_arrivals(rate*arrival_factor,times[-1],phase),r0,q0,
        parameters["concurrency_limit_hypothesis"],running_knots=parameters["state_curve_running_knots"],
        rate_knots=parameters["state_curve_rate_knots"],initial_work_power=power)
    rows=[]
    for t in times:
        row=engine.advance(t)
        row["initial_running_jobs_left_in_model"]=int(np.sum(initial>engine.progress+1e-10))
        row["virtual_progress"]=float(engine.progress)
        rows.append(row)
    return pd.DataFrame(rows)


def exposure_rates(rate):
    rate=np.asarray(rate,float)
    if rate.ndim!=1 or len(rate)==0 or not np.isfinite(rate).all() or np.any(rate<0):
        raise ValueError("Finite nonnegative rates required")
    masks=(rate<2, (rate>=2)&(rate<=30), (rate>30)&(rate<60),
           (rate>=60)&(rate<=100), rate>100)
    assert np.all(np.sum(masks,axis=0)==1)
    return {band:float(mask.mean()) for band,mask in zip(BANDS,masks)}


def size_for_mean_rate(mean_facility_rate,target):
    if not np.isfinite([mean_facility_rate,target]).all() or mean_facility_rate<=0 or target<=0:
        raise ValueError("Positive finite mean and target required")
    return int(np.ceil(mean_facility_rate/target))


def block_rates(counts,seconds):
    counts=np.asarray(counts)
    if counts.ndim!=1 or len(counts)%seconds or seconds<1 or int(seconds)!=seconds:
        raise ValueError("Whole aligned blocks required")
    if not np.isfinite(counts).all() or np.any(counts<0) or np.any(counts!=np.floor(counts)):
        raise ValueError("Nonnegative integer request counts required")
    return counts.reshape(-1,int(seconds)).sum(axis=1)/seconds


def window_summary(obs,rate,parameters,left,right):
    part=obs[(obs.relative_run_s>=left)&(obs.relative_run_s<right)].copy()
    if len(part)<4: raise ValueError("Insufficient state snapshots in diagnostic window")
    stats=diagnose_window(part,rate)
    t=part.replay_s.to_numpy(float); dt=np.diff(t)
    if np.any(dt<=0): raise ValueError("Non-increasing time")
    for field in ("prompt_tokens_per_s","generation_tokens_per_s"):
        stats[field+"_aligned_mean"]=float(np.dot(part[field].to_numpy(float)[1:],dt)/dt.sum())
    for field in ("gpu_kv_cache_pct","prefix_cache_hit_pct"):
        vals=part[field].to_numpy(float)
        stats[field+"_mean"]=float(time_average(t,vals)) if np.isfinite(vals).all() else None
    curve=np.interp(part.running_requests,parameters["state_curve_running_knots"],parameters["state_curve_rate_knots"])
    stats["frozen_curve_at_observed_running_mean"]=float(time_average(t,curve))
    stats["proxy_minus_curve_rps"]=stats["conditional_departure_rps"]-stats["frozen_curve_at_observed_running_mean"]
    stats["proxy_minus_curve_is_causal_error"]=False
    return stats


def selected_errors(obs,pred,left=None,right=None):
    """Score matched snapshots, always excluding the supplied global first state."""
    if len(obs)!=len(pred) or not np.allclose(obs.replay_s,pred.time_s):
        raise ValueError("Aligned observed/predicted timestamps required")
    mask=np.arange(len(obs))>0
    if left is not None: mask &= obs.relative_run_s.to_numpy()>=left
    if right is not None: mask &= obs.relative_run_s.to_numpy()<right
    if not mask.any(): raise ValueError("No eligible snapshots")
    result={"snapshots":int(mask.sum())}
    for state,col in (("running","running_requests"),("waiting","waiting_requests")):
        err=pred[state].to_numpy()[mask]-obs[col].to_numpy()[mask]
        result[state+"_rmse"]=float(np.sqrt(np.mean(err**2)))
        result[state+"_mae"]=float(np.mean(abs(err)))
        result[state+"_bias"]=float(np.mean(err))
    return result


def waiting_onset(obs,pred):
    """First positive SAMPLED Waiting, not exact crossing time."""
    def first(values):
        idx=np.flatnonzero(np.asarray(values)>0)
        return float(obs.relative_run_s.iloc[idx[0]]) if len(idx) else None
    return {"observed_first_sampled_waiting_run_s":first(obs.waiting_requests),
            "predicted_first_sampled_waiting_run_s":first(pred.waiting),
            "observed_waiting_max":int(obs.waiting_requests.max()),
            "predicted_waiting_max":int(pred.waiting.max())}


def interval_diagnostics(obs,rate,parameters):
    """Balance proxy on each logger interval; hidden remaining work prevents direct capacity identification."""
    result=[]
    for i in range(1,len(obs)):
        prev,row=obs.iloc[i-1],obs.iloc[i]
        dt=float(row.replay_s-prev.replay_s)
        if dt<=0: raise ValueError("Non-positive interval")
        delta=float(row.running_requests+row.waiting_requests-prev.running_requests-prev.waiting_requests)
        mu=np.interp([prev.running_requests,row.running_requests],parameters["state_curve_running_knots"],parameters["state_curve_rate_knots"])
        result.append({"relative_start_s":float(prev.relative_run_s),"relative_end_s":float(row.relative_run_s),
            "interval_s":dt,"running_start":int(prev.running_requests),"running_end":int(row.running_requests),
            "waiting_end":int(row.waiting_requests),"delta_unfinished":delta,
            "conditional_departure_rps":rate-delta/dt,"curve_endpoint_average":float(mu.mean()),
            "generation_tokens_per_s":float(row.generation_tokens_per_s),
            "prompt_tokens_per_s":float(row.prompt_tokens_per_s),"source_line":int(row.source_line)})
    return result
