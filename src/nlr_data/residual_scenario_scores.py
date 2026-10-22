"""PL40 ES (24) and VS (25): validation-only, no composite J or auto-selection.

Tan et al., IJEPES 130 (2021) 106955, doi:10.1016/j.ijepes.2021.106955.
Use empirical-distribution scores exactly: M**2 denominator, p=.5, w_ij=1.
Only the target (wind -> centered IT residual) and trajectory length change.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from tqdm.auto import tqdm

from .dynamic_identifiability import _finite_signal, select_manifest_profiles
from .readers import read_timeseries
from .signal_validation import apply_analysis_policy
from .residual_models import PowerProfile, load_condition_targets, fit_m3_by_dataset
from .residual_model_selection import build_parameter_library, simulate_residual, temporal_features

MODELS = ("R0_m3_only", "R1_white_noise", "R2_ou", "R3_jump_ou")


def _inputs(observed, ensemble):
    y, x = np.asarray(observed, float), np.asarray(ensemble, float)
    if y.ndim != 1 or x.ndim != 2 or x.shape[1] != y.size or y.size < 2 or len(x) < 1:
        raise ValueError("Expected y[H], x[M,H], H>=2, M>=1")
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("Nonfinite score inputs")
    return y, x


def energy_score(observed, ensemble):
    """Exact Eq.24, Euclidean norm (NOT squared), diagonal included as zero."""
    y, x = _inputs(observed, ensemble)
    return float(np.linalg.norm(x-y, axis=1).mean() - pdist(x).sum()/len(x)**2)


def block_average_for_scoring(observed, ensemble, *, raw_interval_s, score_interval_s):
    """Aggregate trajectories on a fixed grid before ES/VS evaluation.

    The score equations are unchanged.  This only defines the temporal units
    supplied to them, analogous to the hourly scenario vectors in PL40.
    """
    y, x = _inputs(observed, ensemble)
    ratio = score_interval_s / raw_interval_s
    factor = int(round(ratio))
    if raw_interval_s <= 0 or score_interval_s <= 0 or factor < 1 or not np.isclose(ratio, factor):
        raise ValueError("score_interval_s must be a positive integer multiple of raw_interval_s")
    usable = (y.size // factor) * factor
    if usable < 2 * factor:
        raise ValueError("Scoring grid must contain at least two time steps")
    return (y[:usable].reshape(-1, factor).mean(axis=1),
            x[:, :usable].reshape(len(x), -1, factor).mean(axis=2))


def score_prefixes(observed, ensemble, counts=(25, 50, 100), p=0.5):
    """All time pairs, no lag sampling/downsampling. O(M*H^2), O(H^2) memory.

    Unordered pairs from pdist are multiplied by 2 to recover Eq.25's
    ordered double sum; diagonal terms vanish. Prefixes share random draws.
    """
    y, x = _inputs(observed, ensemble)
    counts = sorted(set(int(n) for n in counts))
    if not counts or counts[0] < 1 or counts[-1] > len(x) or not 0 < p < 2:
        raise ValueError("Invalid ensemble counts or variogram exponent")
    target = pdist(y[:, None])**p
    accumulated = np.zeros_like(target)
    pair_distances = squareform(pdist(x))
    observation_distances = np.linalg.norm(x-y, axis=1)
    scores = {}
    for m, member in enumerate(x, 1):
        accumulated += pdist(member[:, None])**p
        if m in counts:
            scores[m] = {
                "es_kw": float(observation_distances[:m].mean()
                               - pair_distances[:m, :m].sum()/(2*m*m)),
                "vs_kw": float(2*np.square(target-accumulated/m).sum()),
            }
        if m == counts[-1]:
            break
    return scores


def load_train_validation(dataset_root, manifest_path, split_path):
    """Do not open any test signal; preserve the existing audited reader policy."""
    selected = select_manifest_profiles(manifest_path)
    assignment = pd.read_csv(split_path)
    if assignment.groupby("request_rate")["split"].nunique().max() != 1:
        raise ValueError("Same request rate crosses data splits")
    merged = selected.merge(assignment[["condition_id", "split"]],
                            left_on="recommended_condition_id", right_on="condition_id",
                            validate="one_to_one")
    if len(merged) != len(selected):
        raise ValueError("Incomplete split assignment")
    profiles = []
    for _, row in tqdm(merged[merged.split.isin(["train", "validation"])].iterrows(),
                       desc="Loading train/validation (test excluded)"):
        frame = read_timeseries(Path(dataset_root)/row.relative_path,
                                dataset_root=dataset_root, include_metadata_columns=False, validate=True)
        frame = apply_analysis_policy(frame, row)
        t, power, dt = _finite_signal(frame)
        if not np.allclose(np.diff(t), dt, rtol=1e-5, atol=1e-6):
            raise ValueError(f"Nonuniform time grid: {row.profile_id}")
        profiles.append(PowerProfile(str(row.profile_id), str(row.recommended_condition_id),
                                    str(row.prompt_dataset), float(row.request_rate), str(row.split),
                                    t, power, dt, str(row.measurement_origin)))
    return profiles


def condition_scores(rows, draws):
    """Seeds -> profiles -> conditions; MC replicates are not real-data samples."""
    keys = ["prompt_dataset", "request_rate", "condition_id", "profile_id", "model"]
    profile = rows[rows.draws.eq(draws)].groupby(keys, as_index=False)[["es_kw", "vs_kw"]].mean()
    return profile.groupby(keys[:-2]+["model"], as_index=False)[["es_kw", "vs_kw"]].mean()


def paired_intervals(conditions, seed=789, repetitions=2000):
    """Resample request-rate blocks, keeping both prompt datasets together."""
    rng = np.random.default_rng(seed)
    result = []
    for metric in ["es_kw", "vs_kw"]:
        wide = conditions.pivot_table(index="request_rate", columns="model", values=metric)
        for i, a in enumerate(MODELS):
            for b in MODELS[i+1:]:
                delta = (wide[b]-wide[a]).to_numpy()
                boot = delta[rng.integers(len(delta), size=(repetitions, len(delta)))].mean(axis=1)
                result.append(dict(metric=metric, model_a=a, model_b=b,
                                   difference_b_minus_a=float(delta.mean()),
                                   ci95_low=float(np.quantile(boot,.025)),
                                   ci95_high=float(np.quantile(boot,.975)),
                                   rate_blocks=len(delta)))
    return pd.DataFrame(result)


def run_validation(project_root, *, draws=(25,50,100), seeds=(42,43,44),
                   horizon_s=180.0, score_interval_s=1.0, output_dir=None):
    root = Path(project_root).resolve()
    output = Path(output_dir or root/"outputs/residual_es_vs_validation").resolve()
    output.mkdir(parents=True, exist_ok=True)
    counts = sorted(set(draws))
    if not counts or min(counts)<1 or len(set(seeds)) != len(seeds) or not seeds or horizon_s<=0:
        raise ValueError("Invalid run configuration")
    manifest = root/"outputs/condition_audit/audited_run_manifest.csv"
    split = root/"outputs/steady_state_model_comparison/split_assignment.csv"
    target = root/"outputs/graybox_steady_state/online_rate_condition_targets.csv"
    profiles = load_train_validation(root.parent/"NLR_dataset", manifest, split)
    targets = load_condition_targets(target, split)
    m3 = fit_m3_by_dataset(targets, {"train"})
    library = build_parameter_library(profiles, m3, {"train"})
    assert set(library.parameters.split) == {"train"}
    library.parameters.to_csv(output/"train_parameters.csv", index=False)
    validation = [q for q in profiles if q.split == "validation"]
    dt = validation[0].interval_s
    requested_h = round(horizon_s/dt)
    if not np.isclose(requested_h*dt,horizon_s) or requested_h<2:
        raise ValueError("Horizon must be a multiple of sampling interval")
    score_factor = int(round(score_interval_s/dt))
    if score_interval_s <= 0 or not np.isclose(score_factor*dt, score_interval_s):
        raise ValueError("Score interval must be an integer multiple of raw sampling interval")
    for q in validation:
        if not np.isclose(q.interval_s,dt):
            raise ValueError(f"Incompatible grid {q.profile_id}: {q.interval_s}")
    available_h = min(len(q.power_kw) for q in validation)
    h = (min(requested_h, available_h)//score_factor)*score_factor
    if h < 2*score_factor:
        raise ValueError("Insufficient common horizon for the selected scoring interval")
    actual_horizon_s = h*dt
    config = dict(reference="Tan et al. (2021), doi:10.1016/j.ijepes.2021.106955, Eqs.24-25",
                  models=MODELS, draws=counts, seeds=seeds, p=.5, weights="all w_ij=1",
                  requested_horizon_s=horizon_s, actual_horizon_s=actual_horizon_s,
                  samples=h, interval_s=dt,
                  score_interval_s=score_interval_s,
                  score_samples=h//score_factor,
                  segment="first H audited samples, [0,H*dt); fixed block mean only for ES/VS",
                  centering="Observed and every generated trajectory separately demeaned (legacy convention)",
                  scope="Fluctuation-only validation, not forecasting conditional on initial measured state",
                  formula="Original empirical ensemble denominator M^2, not fair-score correction",
                  calibration_splits=["train"], evaluation_split="validation", test_signals_opened=False,
                  n_train=sum(q.split=="train" for q in profiles), n_validation=len(validation),
                  m3={name:asdict(value) for name,value in m3.items()},
                  software=dict(python=platform.python_version(),numpy=np.__version__),
                  hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [manifest,split,target,Path(__file__),root/"src/nlr_data/residual_model_selection.py"]})
    (output/"protocol.json").write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding="utf-8")
    rows, diagnostics, means = [], [], []
    figdir = output/"figures"
    figdir.mkdir(exist_ok=True)
    for q in tqdm(validation, desc="Exact ES/VS validation"):
        power = q.power_kw[:h]
        observed = power-power.mean()
        means.append(dict(profile_id=q.profile_id,condition_id=q.condition_id,
                          prompt_dataset=q.prompt_dataset,request_rate=q.request_rate,
                          observed_mean_kw=float(power.mean()),m3_mean_kw=float(m3[q.prompt_dataset].predict(q.request_rate))))
        base = dict(profile_id=q.profile_id,condition_id=q.condition_id,
                    prompt_dataset=q.prompt_dataset,request_rate=q.request_rate)
        diagnostics.append(dict(**base,model="observed",seed=-1,draw=-1,**temporal_features(observed,dt)))
        plot_examples = {}
        for model in MODELS:
            for seed in seeds:
                seed_key=f"{seed}|{q.profile_id}|{model}".encode()
                rng=np.random.default_rng(int.from_bytes(hashlib.sha256(seed_key).digest()[:8],"little"))
                ensemble=np.stack([simulate_residual(model,length=h,prompt_dataset=q.prompt_dataset,
                                     request_rate=q.request_rate,library=library,rng=rng)
                                   for _ in range(counts[-1])])
                score_observed, score_ensemble = block_average_for_scoring(
                    observed, ensemble, raw_interval_s=dt, score_interval_s=score_interval_s)
                scores=score_prefixes(score_observed,score_ensemble,counts)
                for n, values in scores.items():
                    rows.append(dict(**base,model=model,seed=seed,draws=n,**values))
                # Auxiliary diagnostics use the original 0.1 s signals.  One fixed
                # seed is sufficient; these values are not used for ranking.
                if seed==seeds[0]:
                    for m, member in enumerate(ensemble):
                        diagnostics.append(dict(**base,model=model,seed=seed,draw=m,**temporal_features(member,dt)))
                if seed==seeds[0]:
                    plot_examples[model]=ensemble
            # Incremental outputs survive interruption without marking completion.
            pd.DataFrame(rows).to_csv(output/"scores_by_profile_seed.csv",index=False)
        # First validation profile for each prompt, chosen without consulting scores.
        if q == next(v for v in validation if v.prompt_dataset==q.prompt_dataset):
            fig,axes=plt.subplots(4,1,figsize=(12,9),sharex=True)
            for ax,model in zip(axes,MODELS):
                ens=plot_examples[model]
                ax.fill_between(np.arange(h)*dt,*np.quantile(ens,[.05,.95],axis=0),alpha=.25,label="pointwise 5-95% (not simultaneous)")
                ax.plot(np.arange(h)*dt,ens[0],lw=.5,label="first draw (not best match)")
                ax.plot(np.arange(h)*dt,observed,color="black",lw=.6,label="observed")
                ax.set_ylabel("Residual (kW)"); ax.set_title(model)
            axes[0].legend(fontsize=7); axes[-1].set_xlabel("Time (s)")
            fig.suptitle(f"{q.prompt_dataset} | {q.request_rate:g} requests/s")
            fig.tight_layout(); fig.savefig(figdir/f"example_{q.prompt_dataset}.png",dpi=140); plt.close(fig)
    scores=pd.DataFrame(rows)
    diagnostics=pd.DataFrame(diagnostics)
    diagnostics.to_csv(output/"diagnostics_by_draw.csv",index=False)
    mean_table=pd.DataFrame(means)
    mean_table["mean_bias_kw"]=mean_table.m3_mean_kw-mean_table.observed_mean_kw
    mean_table.to_csv(output/"m3_mean_bias_separate.csv",index=False)
    cond=condition_scores(scores,counts[-1]); cond.to_csv(output/"scores_by_condition.csv",index=False)
    summary=cond.groupby("model")[["es_kw","vs_kw"]].mean().reindex(MODELS)
    summary.to_csv(output/"validation_summary.csv")
    cond.groupby(["prompt_dataset","model"])[["es_kw","vs_kw"]].mean().to_csv(output/"summary_by_prompt.csv")
    uncertainty=paired_intervals(cond); uncertainty.to_csv(output/"paired_rate_block_bootstrap.csv",index=False)
    sensitivity=scores.groupby(["draws","seed","model"])[["es_kw","vs_kw"]].mean().reset_index()
    sensitivity.to_csv(output/"ensemble_size_seed_sensitivity.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric,title in zip(axes,["es_kw","vs_kw"],["Energy score (kW), lower better","Variogram score p=0.5 (kW), lower better"]):
        ax.bar([m[:2] for m in MODELS],summary[metric]); ax.set_title(title); ax.grid(axis="y",alpha=.2)
    fig.tight_layout(); fig.savefig(figdir/"validation_es_vs.png",dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric in zip(axes,["es_kw","vs_kw"]):
        for model in MODELS:
            part=sensitivity[sensitivity.model.eq(model)].groupby("draws")[metric]
            ax.errorbar(part.mean().index,part.mean(),yerr=part.std().fillna(0),marker="o",label=model[:2])
        ax.set_title(metric+" (mean +/- seed SD)"); ax.set_xlabel("ensemble size"); ax.legend()
    fig.tight_layout(); fig.savefig(figdir/"ensemble_sensitivity.png",dpi=160); plt.close(fig)
    status=dict(status="validation_complete_no_family_locked",es_best=summary.es_kw.idxmin(),
                vs_best=summary.vs_kw.idxmin(),test_evaluated=False,composite_score=None,
                next_step="Review ES/VS conflicts, seed/ensemble sensitivity and diagnostics before locking protocol for test")
    (output/"status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    report=f"""# R0-R3 ES/VS validation 평가

- 근거: Paper list [40], Tan et al., IJEPES 130 (2021) 106955, 식 (24), (25).
- 기존 모델 및 J 결과는 수정하지 않음. ES/VS를 합산하지 않고 최종 후보를 자동 확정하지 않음.
- Train {config['n_train']}개 profile만 M3/OU/Jump-OU 파라미터 보정에 사용.
- Validation {len(validation)}개 profile, 요청 구간 {horizon_s:g}초 중 모든 조건에 공통인 첫 {actual_horizon_s:g}초 사용, 원자료 H={h}, dt={dt:.3g}초. Test signal 미열람/미평가.
- ES/VS 입력은 고정된 {score_interval_s:g}초 block mean(H={h//score_factor})으로 구성. 식 자체는 원문과 동일하며, 0.1초 원자료는 파형·보조 진단에 유지.
- 시나리오 수 {counts}, seed {list(seeds)}. 같은 seed 내 prefix를 사용. 시나리오 수 효과에는 원식의 유한 ensemble 효과도 포함됨.
- ES: 원문 M^2 분모/Euclidean norm. VS: p=0.5, 모든 시점 쌍, w=1, 원문 double sum 그대로.
- 풍력에서 평균 제거 IT residual로 평가대상만 변경. 평균 제거는 기존 구현 유지: raw OU의 무조건부 과정 자체가 아닌 유한 구간 centered fluctuation의 평가.
- M3 평균 편향은 m3_mean_bias_separate.csv로 분리. 본 결과는 전체 IT 전력/DOE/DFL 성능 검증이 아님.

## Validation 평균 결과

```text
{summary.to_string()}
```

![ES/VS](figures/validation_es_vs.png)
![Ensemble sensitivity](figures/ensemble_sensitivity.png)

- ES 최소: {status['es_best']}; VS 최소: {status['vs_best']}.
- 두 점수 순위가 다르거나 paired CI가 0을 포함하면 단일 모델 우수성을 확정하지 않음.
- paired_rate_block_bootstrap.csv: model_b - model_a (음수면 b 우수). seed 평균 후 request-rate block을 resampling하여 두 prompt의 관계 유지. 조건 모집단에 대한 탐색적 bootstrap이며 독립 실험 반복이 아님.
- diagnostics_by_draw.csv: 표준편차/분위수/ACF/증분/PSD 등 보조 진단. 진단값으로 새 J를 만들지 않음. 영분산 R0의 상관계수는 물리적으로 정의되지 않으므로 legacy zero placeholder를 해석하지 말 것.
- 대표 그림은 prompt별 첫 validation profile, 첫 seed/첫 draw로 고정 선택. 임의 best-match 파형을 선택하지 않음.

## 다음 단계

1. 점수의 ensemble/seed 민감도와 condition별 이질성, 분포/ACF/증분을 검토.
2. 평가 protocol 및 후보를 확정한 뒤 별도의 test 실행. 현재 test 결과를 사용해 튜닝하지 않음.
3. M3+residual 전체 IT profile은 평균 편향을 포함해 별도 검증.
"""
    (output/"report_ko.md").write_text(report,encoding="utf-8")
    return status
