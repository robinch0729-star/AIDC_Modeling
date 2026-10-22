"""Small five-minute IT operating baseline using frozen M3/R3 artifacts.

No calibration is performed. NLR/Azure transfer, fixed equal-intensity routing,
peak-based sizing and the optional state/power bridge are scenario assumptions.
R3 aggregation uses an exact algebraic sum under conditional independence and
identical coefficients within each service (not a Gaussian CLT approximation).
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from .carryover_replay import StateReplay

DATASETS = {"code": "InstructCoder", "conversation": "mlperf-inference-llama2-data"}
PARAM_COLUMNS = ["phi", "jump_evidence", "innovation_std_kw"] + [
    f"{role}_{field}" for role in ("negative", "background", "positive")
    for field in ("weight", "mean_kw", "std_kw")]


def m3(rate, p):
    rate = np.asarray(rate, float)
    if not np.isfinite(rate).all() or np.any(rate < 0):
        raise ValueError("Finite nonnegative offered rate required")
    z = np.clip(p["k"] * (np.log1p(rate) - p["x0"]), -700, 700)
    return p["p0_kw"] + p["delta_p_kw"] / (1 + np.exp(-z))


def block_mean(values, seconds=300, offset=0):
    x = np.asarray(values, float)
    if x.ndim != 1 or seconds < 1 or int(seconds) != seconds or not 0 <= offset < seconds:
        raise ValueError("One-second values and a valid integer block required")
    n = (len(x) - offset) // seconds
    if n < 1:
        raise ValueError("No complete block")
    return x[offset:offset+n*seconds].reshape(n, seconds).mean(axis=1)


def choose_design(source_counts, parameters, target_mw, peak_rate=30., parent=60, max_multiplier=10000):
    """Choose integer demand multiplier and counts ONCE, before stochastic output.

    Full observed period used for retrospective scenario sizing, not forecasting.
    Integer rounding prevents an exactly prescribed mean; output is never scaled.
    """
    rates = {s: block_mean(x, parent) for s, x in source_counts.items()}
    def evaluate(mult):
        nodes = {s: int(math.ceil(mult * x.max() / peak_rate)) for s, x in rates.items()}
        power = sum(nodes[s] * m3(x * mult / nodes[s], parameters[s]) for s, x in rates.items())/1000
        return float(power.mean()), nodes
    lo, hi = 1, int(max_multiplier)
    if not evaluate(lo)[0] <= target_mw <= evaluate(hi)[0]:
        raise ValueError("Target outside design search bracket")
    while hi-lo > 1:
        mid = (lo+hi)//2
        if evaluate(mid)[0] < target_mw:
            lo = mid
        else:
            hi = mid
    k = min((lo,hi), key=lambda v: abs(evaluate(v)[0]-target_mw))
    mean, nodes = evaluate(k)
    return {"request_multiplier": k, "instance_counts": nodes,
            "target_it_mean_mw": target_mw, "design_expected_mean_mw": mean,
            "design_mean_error_mw": mean-target_mw,
            "parent_interval_s": parent, "peak_parent_rate_rps": peak_rate,
            "source_mean_rps": {s:float(x.mean()) for s,x in rates.items()},
            "source_peak_parent_rps": {s:float(x.max()) for s,x in rates.items()},
            "source_mix": {s:float(x.sum()/sum(y.sum() for y in source_counts.values())) for s,x in source_counts.items()},
            "instance_mix": {s:n/sum(nodes.values()) for s,n in nodes.items()}}


def locked_coefficients(library, dataset, rate):
    """Exact eight-neighbour weighting as in ResidualParameterLibrary.

    Rates below one are retained for M3 but use the one-rps residual boundary.
    Upper extrapolation above 100 is rejected. The legacy 81-100 benchmark is
    not called independent validation or an SLO capacity certificate here.
    """
    rate = np.asarray(rate, float)
    if rate.ndim != 1 or not np.isfinite(rate).all() or np.any(rate < 0) or np.any(rate > 100):
        raise ValueError("Residual scenario support is 0..100 rps with flagged low extension")
    table = library[library.prompt_dataset.eq(dataset)].sort_values("request_rate", kind="stable")
    if len(table) < 8:
        raise ValueError("Eight frozen neighbours required")
    rr = table.request_rate.to_numpy(float)
    vv = table[PARAM_COLUMNS].to_numpy(float)
    if not np.isfinite(vv).all():
        raise ValueError("Missing frozen coefficient")
    output = np.empty((len(rate), len(PARAM_COLUMNS)))
    for start in range(0, len(rate), 4096):
        r = np.maximum(rate[start:start+4096], 1.)
        distance = np.abs(r[:,None]-rr[None,:])
        index = np.argsort(distance, axis=1, kind="stable")[:,:8]
        weight = 1/np.maximum(np.take_along_axis(distance,index,axis=1), .5)
        weight /= weight.sum(axis=1,keepdims=True)
        output[start:start+len(r)] = np.sum(vv[index] * weight[:,:,None],axis=1)
    phi = np.clip(output[:,0],0,.9995)
    gate = np.clip(output[:,1],0,1)
    sigma = np.maximum(output[:,2],1e-12)
    weight = np.maximum(output[:,[3,6,9]],1e-8)
    weight /= weight.sum(axis=1,keepdims=True)
    means = output[:,[4,7,10]]
    means -= np.sum(means*weight,axis=1,keepdims=True)
    std = np.maximum(output[:,[5,8,11]],1e-12)
    # Four-component marginal innovation: non-jump + three gated components.
    return {"phi":phi, "prob":np.c_[1-gate,gate[:,None]*weight],
            "mean":np.c_[np.zeros(len(rate)),means],
            "std":np.c_[sigma,std]}


def aggregate_innovation(rng, n, prob, mean, std):
    """Exact sum distribution of n IID Gaussian-mixture innovations.

    C~Multinomial(n,p), sum eta | C ~ Normal(sum C*m, sum C*sigma^2).
    Sequential binomials permit different p at every native sample.
    """
    prob, mean, std = map(lambda x:np.asarray(x,float), (prob,mean,std))
    if n < 1 or int(n) != n or prob.shape != mean.shape or prob.shape != std.shape or prob.shape[-1] != 4:
        raise ValueError("Positive integer population and four matching components required")
    if np.any(prob < 0) or not np.allclose(prob.sum(axis=-1),1):
        raise ValueError("Invalid component probabilities")
    remaining = np.full(prob.shape[:-1],n,dtype=np.int64)
    remaining_p = np.ones(prob.shape[:-1])
    counts = np.empty(prob.shape,dtype=np.int64)
    for j in range(3):
        conditional = np.divide(prob[...,j],remaining_p,out=np.zeros_like(remaining_p),where=remaining_p>1e-14)
        counts[...,j] = rng.binomial(remaining,np.clip(conditional,0,1))
        remaining -= counts[...,j]
        remaining_p -= prob[...,j]
    counts[...,3] = remaining
    location = np.sum(counts*mean,axis=-1)
    variance = np.sum(counts*std**2,axis=-1)
    return rng.normal(location,np.sqrt(variance))


def sum_residual_seconds(coeff, n, seed, burn_seconds=120, native_dt=.1):
    """Return exact one-second means of a native-.1-s summed R3 path.

    Zero initial state followed by discarded burn-in. No state resets or
    finite-window demeaning. Conditional independence is an assumption.
    """
    if not np.isclose(native_dt,.1):
        raise ValueError("Frozen R3 requires 0.1-second recurrence")
    rng = np.random.default_rng(seed)
    burn = int(round(burn_seconds/native_dt))
    state = 0.
    eta = aggregate_innovation(rng,n,np.tile(coeff["prob"][0],(burn,1)),
        np.tile(coeff["mean"][0],(burn,1)),np.tile(coeff["std"][0],(burn,1)))
    for noise in eta:
        state = coeff["phi"][0]*state+noise
    result = np.empty(len(coeff["phi"]))
    peak_abs = 0.
    for a in range(0,len(result),3600):
        b = min(a+3600,len(result))
        phi = np.repeat(coeff["phi"][a:b],10)
        eta = aggregate_innovation(rng,n,*[np.repeat(coeff[key][a:b],10,axis=0) for key in ("prob","mean","std")])
        values = np.empty(len(eta))
        for i in range(len(values)):
            state = phi[i]*state+eta[i]
            values[i] = state
        result[a:b] = values.reshape(-1,10).mean(axis=1)
        peak_abs = max(peak_abs,float(np.max(np.abs(values))))
    return result,{"native_dt_s":native_dt,"native_samples":len(result)*10,
                   "summed_native_residual_max_abs_kw":peak_abs,"finite_window_demeaning":False}


def regular_variable_arrivals(rates, phase=.5):
    """Representative instance: integrate piecewise 1-s offered intensity.

    A deterministic phase discretizes expected requests; no claim that these
    are actual Azure per-instance events. Total differs from intensity by <1.
    """
    r = np.asarray(rates,float)
    if r.ndim != 1 or not np.isfinite(r).all() or np.any(r < 0) or not 0 < phase <= 1:
        raise ValueError("Invalid arrival intensity")
    cumulative = np.r_[0.,np.cumsum(r)]
    levels = np.arange(phase,cumulative[-1]+1e-10,1.)
    bins = np.searchsorted(cumulative,levels,side="right")-1
    bins = np.minimum(bins,len(r)-1)
    # Exact boundary levels followed by zero bins resolve to the next positive bin.
    valid = r[bins] > 0
    return bins[valid] + (levels[valid]-cumulative[bins[valid]])/r[bins[valid]]


def carryover_bridge(rates, parameters, power_parameters, warmup_s=600, sample_dt=.1):
    """Frozen event-driven processing model + HYPOTHETICAL equilibrium bridge.

    r_eq=mu_hat(R) (mean normalized work=1) is a steady-equivalent offered
    rate, NOT measured completion throughput. Original M3 remains a distinct
    offered-rate baseline. A memoryless bridge control separates saturation
    mapping offset from memory. No R3 is included in this structural comparison.
    """
    r = np.asarray(rates,float)
    if r.ndim != 1 or len(r)==0 or np.any(r<0) or np.any(r>100):
        raise ValueError("Scenario rates outside 0..100")
    warm = np.full(warmup_s,r[0])
    all_rates = np.r_[warm,r]
    arrivals = regular_variable_arrivals(all_rates)
    engine = StateReplay(arrivals,0,0,parameters["concurrency_limit_hypothesis"],
        running_knots=parameters["state_curve_running_knots"],rate_knots=parameters["state_curve_rate_knots"])
    initial = engine.advance(warmup_s)
    count = int(round(1/sample_dt))
    if not np.isclose(count*sample_dt,1):
        raise ValueError("Sample interval must divide one second")
    running=np.empty(len(r)); waiting=np.empty(len(r)); power=np.empty(len(r)); exits=np.empty(len(r),int)
    req_equiv=np.empty(len(r)); previous=initial["completions"]
    max_balance=0
    # Midpoint quadrature only samples the event-driven process; it does not
    # change admission/completion times. State persists across five-minute bins.
    for second in range(len(r)):
        rr=[]; qq=[]
        for j in range(count):
            snapshot=engine.advance(warmup_s+second+(j+.5)*sample_dt)
            rr.append(snapshot["running"]); qq.append(snapshot["waiting"])
            max_balance=max(max_balance,abs(snapshot["conservation_error"]))
        equivalent=np.interp(rr,parameters["state_curve_running_knots"],parameters["state_curve_rate_knots"])
        running[second]=np.mean(rr); waiting[second]=np.mean(qq)
        req_equiv[second]=np.mean(equivalent)
        power[second]=np.mean(m3(equivalent,power_parameters))
        end=engine.advance(warmup_s+second+1.)
        exits[second]=end["completions"]-previous; previous=end["completions"]
    original=m3(r,power_parameters)
    memoryless=m3(np.minimum(r,max(parameters["state_curve_rate_knots"])),power_parameters)
    return {"offered_rate_rps":r,"running_mean":running,"waiting_mean":waiting,
        "steady_equivalent_rate_rps":req_equiv,"completions_count":exits,
        "original_kw":original,"memoryless_bridge_kw":memoryless,"stateful_bridge_kw":power}, {
        "warmup_seconds":warmup_s,"warmup_policy":"constant first rate, empty start, continuous state",
        "initial_state_after_warmup":initial,"final_state":engine.snapshot(),
        "conservation_max_error_requests":max_balance,"power_sampling_dt_s":sample_dt,
        "expected_arrivals_with_warmup":float(all_rates.sum()),"generated_arrivals":len(arrivals),
        "mapping_validated":False,"other_exits_assumed_zero":True}


def paired_metrics(original, comparator, block_s=300, offset=0):
    a,b=block_mean(original,block_s,offset),block_mean(comparator,block_s,offset)
    delta=b-a
    return {"block_s":block_s,"offset_s":offset,"complete_bins":len(a),
        "dropped_edge_seconds":len(original)-len(a)*block_s,
        "mean_difference_mw":float(delta.mean()),"mae_mw":float(np.abs(delta).mean()),
        "max_abs_difference_mw":float(np.abs(delta).max()),
        "mae_pct_of_reference_mean":float(100*np.abs(delta).mean()/a.mean()),
        "max_abs_pct_of_reference_mean":float(100*np.abs(delta).max()/a.mean()),
        "reference_energy_mwh":float(a.sum()*block_s/3600),
        "difference_energy_mwh":float(delta.sum()*block_s/3600),
        "absolute_bin_energy_difference_mwh":float(np.abs(delta).sum()*block_s/3600)}
