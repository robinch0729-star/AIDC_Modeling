"""Conditional NLR state replay, not a power model or a vLLM emulator.

F1/F2 simplify the lifetime/FIFO scaffold in Wilkins et al. (2026), Sec. 3.3.
S1 implements the fixed-concurrency Sd-LPS service rule in Gupta & Zhang (2022),
Sec. 2.1 Eqs. (2)-(3). Its NLR rate curve and unit job sizes are OUR assumptions.
"""
from __future__ import annotations
from collections import deque
import heapq
import math
import numpy as np
import pandas as pd
from .nlr_processing_diagnostic import SERVICES, linear_slope, time_average, diagnose_window

MODEL_IDS = ("B0_persistence", "F1_short_lifetime", "F2_capacity_lifetime", "S1_state_dependent_lps")
EPS = 1e-9


def make_segments(runs: pd.DataFrame, raw: pd.DataFrame, left=60., right=240.) -> dict:
    raw = raw.copy()
    raw["timestamp"] = pd.to_datetime(raw.timestamp)
    result = {}
    for row in runs.to_dict("records"):
        start = pd.Timestamp(row["run_start"])
        rel = (raw.timestamp-start).dt.total_seconds()
        part = raw[(rel >= left) & (rel < right)].copy()
        if len(part) < 6 or part.timestamp.duplicated().any():
            raise ValueError("Insufficient or duplicate state snapshots")
        part["relative_run_s"] = (part.timestamp-start).dt.total_seconds()
        part["replay_s"] = (part.timestamp-part.timestamp.iloc[0]).dt.total_seconds()
        rate = float(row["request_rate"])
        service = row["dataset-path"].rsplit("/", 1)[-1]
        if service not in SERVICES or not rate.is_integer():
            raise ValueError("Unsupported service/rate")
        result[str(row["id"])] = {"run_id": str(row["id"]), "service": service,
            "rate": rate, "split": "calibration" if int(rate)%2 else "comparison",
            "observations": part}
    return result


def isotonic(values, weights=None):
    """Pooled adjacent violators, used only as explicit monotone curve regularization."""
    values = np.asarray(values, float)
    weights = np.ones(len(values)) if weights is None else np.asarray(weights, float)
    if len(values) != len(weights) or not np.isfinite(values).all() or np.any(weights <= 0):
        raise ValueError("Invalid isotonic input")
    blocks = []
    for i, (v,w) in enumerate(zip(values,weights)):
        blocks.append([i, i+1, float(w), float(v*w)])
        while len(blocks)>1 and blocks[-2][3]/blocks[-2][2] > blocks[-1][3]/blocks[-1][2]:
            b, a = blocks.pop(), blocks.pop()
            blocks.append([a[0],b[1],a[2]+b[2],a[3]+b[3]])
    output = np.empty(len(values))
    for left,right,w,v in blocks:
        output[left:right] = v/w
    return output


def fit_parameters(segments: dict) -> tuple[dict, list]:
    """Uses odd-rate runs and 60-150 s snapshots ONLY; no full-audit parameters."""
    rows = []
    for segment in segments.values():
        if segment["split"] != "calibration":
            continue
        part = segment["observations"]
        part = part[part.relative_run_s < 150]
        stats = diagnose_window(part, segment["rate"])
        rows.append({"run_id": segment["run_id"], "service": segment["service"], **stats})
    table = pd.DataFrame(rows)
    result = {}
    for service in SERVICES:
        t = table[table.service.eq(service)].copy()
        anchors = t[t.configured_rps.between(10,30) & t.low_drift_screen_2pct]
        high = t[t.configured_rps.ge(60) & t.waiting_positive_all_snapshots
                 & t.metric_count.ge(6) & t.max_metric_gap_s.le(20)]
        if anchors.empty or high.empty:
            raise ValueError("Required calibration regime missing")
        lifetime = float(anchors.running_over_configured_rate_s.median())
        mu = float(high.conditional_departure_rps.median())
        k = int(t.running_max.max())
        # A single high-load anchor avoids pseudo-replication of many saturated runs.
        xh = float(high.running_mean.median())
        low = t[t.configured_rps.lt(60) & t.running_mean.lt(xh)].sort_values("running_mean")
        grouped = low.groupby("running_mean", as_index=False).conditional_departure_rps.median()
        xs = np.r_[0., grouped.running_mean.to_numpy(), xh, float(k)]
        raw_y = np.r_[0., np.clip(grouped.conditional_departure_rps.to_numpy(), 0., mu), mu, mu]
        ys = isotonic(raw_y)
        if np.any(np.diff(xs)<=0) or mu <= 0 or lifetime <= 0:
            raise ValueError("Invalid calibrated curve")
        result[service] = {"training_run_ids": t.run_id.tolist(),
            "training_window_s": [60,150], "concurrency_limit_hypothesis": k,
            "concurrency_limit_is_confirmed_setting": False,
            "short_lifetime_s": lifetime, "conditional_high_load_rate_rps": mu,
            "capacity_matched_lifetime_s": k/mu,
            "state_curve_running_knots": xs.tolist(), "state_curve_rate_knots": ys.tolist(),
            "state_curve_raw_rate_knots": raw_y.tolist(),
            "curve_method": "monotone pooled-adjacent-violators on conditional rate vs mean Running, then linear interpolation",
            "curve_is_literature_formula": False, "unit_job_work": 1.,
            "mean_work_unit_status": "normalized research assumption, not measured token work",
            "rate_is_observed_completion": False}
    return result, rows


class StateReplay:
    """Event-driven FIFO admission with equal progress for all active jobs.

    F models: progress speed=1/lifetime. S1: speed=mu(R)/R. Initial request
    residual work is unobserved, so uniform quantiles on (0,1) are an assumption.
    Sampling does not reset the queue or consume future requests.
    """
    def __init__(self, arrivals, running0, waiting0, limit, *, lifetime=None,
                 running_knots=None, rate_knots=None, initial_work_power=1.):
        self.arrivals = np.asarray(arrivals, float)
        if np.any(~np.isfinite(self.arrivals)) or np.any(self.arrivals <= 0) or np.any(np.diff(self.arrivals)<0):
            raise ValueError("Arrival times must be finite, positive and sorted")
        if any(int(v)!=v or v<0 for v in (running0,waiting0,limit)) or limit < 1 or running0 > limit:
            raise ValueError("Invalid initial state or concurrency limit")
        if not np.isfinite(initial_work_power) or initial_work_power <= 0:
            raise ValueError("Invalid initial-work shape")
        if (lifetime is None) == (running_knots is None):
            raise ValueError("Choose either lifetime or state-dependent rate")
        self.limit, self.waiting = int(limit), int(waiting0)
        self.initial_total = int(running0+waiting0)
        self.heap = [float(((i+.5)/running0)**initial_work_power) for i in range(int(running0))]
        heapq.heapify(self.heap)
        self.lifetime = lifetime
        if lifetime is not None and (not np.isfinite(lifetime) or lifetime <= 0):
            raise ValueError("Positive lifetime required")
        self.running_knots = np.asarray(running_knots, float) if running_knots is not None else None
        self.rate_knots = np.asarray(rate_knots, float) if rate_knots is not None else None
        if self.running_knots is not None:
            if len(self.running_knots)<2 or len(self.running_knots)!=len(self.rate_knots):
                raise ValueError("Invalid rate curve")
            if not np.isfinite(self.running_knots).all() or not np.isfinite(self.rate_knots).all():
                raise ValueError("Non-finite curve")
            if np.any(np.diff(self.running_knots)<=0) or np.any(self.rate_knots<0):
                raise ValueError("Invalid rate curve ordering or sign")
        self.clock = self.progress = 0.
        self.arrived = self.completed = self.started = 0
        self.event_count = 0

    def _admit(self):
        count = min(self.waiting, self.limit-len(self.heap))
        for _ in range(count):
            heapq.heappush(self.heap,self.progress+1.)
        self.waiting -= count
        self.started += count

    def _speed(self):
        n=len(self.heap)
        if not n: return 0.
        if self.lifetime is not None: return 1./self.lifetime
        return float(np.interp(n,self.running_knots,self.rate_knots))/n

    def advance(self, target):
        if not np.isfinite(target) or target < self.clock-EPS:
            raise ValueError("Cannot replay backwards")
        if target == 0:
            return self.snapshot()
        self._admit()
        while self.clock < target-EPS:
            speed = self._speed()
            next_a = self.arrivals[self.arrived] if self.arrived<len(self.arrivals) else math.inf
            next_c = self.clock+max(0.,self.heap[0]-self.progress)/speed if speed>0 else math.inf
            event = min(float(next_a),next_c,float(target))
            if event < self.clock-EPS:
                raise RuntimeError("Noncausal event")
            self.progress += speed*(event-self.clock)
            self.clock = event
            # Process simultaneous completion before arrivals. No request is lost.
            while self.heap and self.heap[0] <= self.progress+1e-10:
                heapq.heappop(self.heap)
                self.completed += 1
            self._admit()
            while self.arrived<len(self.arrivals) and self.arrivals[self.arrived] <= event+EPS:
                self.waiting += 1
                self.arrived += 1
            self._admit()
            self.event_count += 1
        self.clock = float(target)
        row = self.snapshot()
        if row["conservation_error"] != 0:
            raise AssertionError("Request count conservation failed")
        return row

    def snapshot(self):
        return {"time_s":self.clock,"running":len(self.heap),"waiting":self.waiting,
            "arrivals":self.arrived,"completions":self.completed,"starts":self.started,
            "unfinished":len(self.heap)+self.waiting,
            "conservation_error":len(self.heap)+self.waiting-self.initial_total-self.arrived+self.completed}


def regular_arrivals(rate, duration, phase=.5):
    if not np.isfinite(rate) or rate<=0 or duration<0 or not 0<phase<=1:
        raise ValueError("Invalid arrival configuration")
    count=max(0,int(math.floor(duration*rate-phase+1e-8))+1)
    return (np.arange(count,dtype=float)+phase)/rate


def simulate(model, parameters, rate, times, running0, waiting0, *, phase=.5, work_power=1.):
    times=np.asarray(times,float)
    if len(times)==0 or times[0] != 0 or np.any(np.diff(times)<=0):
        raise ValueError("Output times must begin at 0 and be strictly increasing")
    if model=="B0_persistence":
        return pd.DataFrame({"time_s":times,"running":running0,"waiting":waiting0,
                             "unfinished":running0+waiting0})
    arrivals=regular_arrivals(rate,times[-1],phase)
    kwargs={"initial_work_power":work_power}
    if model=="F1_short_lifetime": kwargs["lifetime"]=parameters["short_lifetime_s"]
    elif model=="F2_capacity_lifetime": kwargs["lifetime"]=parameters["capacity_matched_lifetime_s"]
    elif model=="S1_state_dependent_lps":
        kwargs.update(running_knots=parameters["state_curve_running_knots"],rate_knots=parameters["state_curve_rate_knots"])
    else: raise ValueError(f"Unknown model {model}")
    engine=StateReplay(arrivals,running0,waiting0,parameters["concurrency_limit_hypothesis"],**kwargs)
    return pd.DataFrame([engine.advance(t) for t in times])


def score(observed, prediction):
    """Exclude the supplied initial observation from errors; no composite score."""
    if len(observed)!=len(prediction) or len(observed)<3:
        raise ValueError("Aligned state trajectories required")
    t=observed.replay_s.to_numpy(float)
    if not np.allclose(t,prediction.time_s): raise ValueError("Timestamp mismatch")
    output={"scored_snapshots":len(t)-1,"horizon_s":float(t[-1])}
    for state, column in (("running","running_requests"),("waiting","waiting_requests")):
        actual=observed[column].to_numpy(float)
        pred=prediction[state].to_numpy(float)
        error=pred[1:]-actual[1:]
        output.update({f"{state}_rmse":float(np.sqrt(np.mean(error**2))),
            f"{state}_mae":float(np.mean(abs(error))), f"{state}_bias":float(np.mean(error)),
            f"{state}_endpoint_error":float(error[-1]),
            f"{state}_slope_error_rps":float(linear_slope(t,pred)-linear_slope(t,actual))})
    b=observed.running_requests.to_numpy(float)+observed.waiting_requests.to_numpy(float)
    error=prediction.unfinished.to_numpy(float)[1:]-b[1:]
    output["unfinished_rmse"]=float(np.sqrt(np.mean(error**2)))
    output["max_waiting_observed"]=int(observed.waiting_requests.max())
    output["max_waiting_predicted"]=int(prediction.waiting.max())
    return output
