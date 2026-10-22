import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from nlr_data.residual_scenario_scores import (
    energy_score, score_prefixes, condition_scores, block_average_for_scoring,
)


def test_matches_literal_paper_double_sums():
    rng=np.random.default_rng(2)
    y=rng.normal(size=7); x=rng.normal(size=(8,7))
    got=score_prefixes(y,x,[1,4,8])
    for m in [1,4,8]:
        es=sum(np.linalg.norm(y-z) for z in x[:m])/m
        es-=sum(np.linalg.norm(a-b) for a in x[:m] for b in x[:m])/(2*m*m)
        vs=sum((abs(y[i]-y[j])**.5-np.mean(abs(x[:m,i]-x[:m,j])**.5))**2
               for i in range(7) for j in range(7))
        assert got[m]["es_kw"]==pytest.approx(es)
        assert got[m]["vs_kw"]==pytest.approx(vs)
        assert energy_score(y,x[:m])==pytest.approx(es)


def test_r0_and_identical_trajectories():
    y=np.array([-.2,.1,.1])
    exact=np.tile(y,(5,1))
    assert score_prefixes(y,exact,[5])[5]==pytest.approx(dict(es_kw=0,vs_kw=0))
    zero=np.zeros((5,3))
    assert energy_score(y,zero)==pytest.approx(np.linalg.norm(y))
    assert np.isfinite(score_prefixes(y,zero,[5])[5]["vs_kw"])


def test_common_shift_and_scale():
    rng=np.random.default_rng(3); y=rng.normal(size=5); x=rng.normal(size=(6,5))
    base=score_prefixes(y,x,[6])[6]
    assert score_prefixes(y+10,x+10,[6])[6]==pytest.approx(base)
    scaled=score_prefixes(y*3,x*3,[6])[6]
    assert scaled==pytest.approx({k:v*3 for k,v in base.items()})


def test_invalid_inputs():
    with pytest.raises(ValueError): score_prefixes([0,1],[[0,np.nan]],[1])
    with pytest.raises(ValueError): score_prefixes([0,1],[[0,1]],[2])


def test_fixed_block_average_for_scoring():
    y=np.array([0.,2.,4.,6.]); x=np.array([[0.,2.,4.,6.],[2.,4.,6.,8.]])
    y_agg,x_agg=block_average_for_scoring(y,x,raw_interval_s=.5,score_interval_s=1.)
    assert np.array_equal(y_agg,[1.,5.])
    assert np.array_equal(x_agg,[[1.,5.],[3.,7.]])
    with pytest.raises(ValueError):
        block_average_for_scoring(y,x,raw_interval_s=.3,score_interval_s=1.)


def test_condition_aggregation_does_not_overweight_repeats():
    rows=[]
    for cond,profiles in [("a",[1,2]),("b",[3])]:
        for profile in profiles:
            for seed in [0,1]:
                rows.append(dict(prompt_dataset="d",request_rate=1 if cond=="a" else 2,
                                 condition_id=cond,profile_id=profile,model="R0",draws=10,seed=seed,
                                 es_kw=2 if cond=="a" else 8,vs_kw=4 if cond=="a" else 16))
    result=condition_scores(pd.DataFrame(rows),10)
    assert result.es_kw.mean()==5
