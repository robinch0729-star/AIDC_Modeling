"""Stage 4-2: fit on odd rates, condition on initial stocks, replay even rates."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from aidc_facility.carryover_replay import MODEL_IDS, make_segments, fit_parameters, simulate, score
from aidc_facility.nlr_processing_diagnostic import SERVICES, sha256

LABELS={"B0_persistence":"B0 hold initial state", "F1_short_lifetime":"F1 short lifetime",
        "F2_capacity_lifetime":"F2 capacity lifetime", "S1_state_dependent_lps":"S1 state-dependent LPS"}
SHORT={SERVICES[0]:"InstructCoder", SERVICES[1]:"Llama2-data"}


def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(np.bool_,)): return bool(value)
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def dump(path,value):
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def group_scores(scores):
    metrics=["running_rmse","waiting_rmse","unfinished_rmse","running_mae","waiting_mae",
             "running_bias","waiting_bias","running_slope_error_rps","waiting_slope_error_rps"]
    by_band=scores.groupby(["service","band","model"],sort=False)[metrics].mean().reset_index()
    by_band["run_count"]=scores.groupby(["service","band","model"],sort=False).size().to_numpy()
    overall=scores.groupby(["service","model"],sort=False)[metrics].mean().reset_index()
    return by_band,overall


def plot(path,title,subtitle,panels,footer):
    """Local scientific plots; keep earlier diagnostic scripts unchanged."""
    from PIL import Image, ImageDraw, ImageFont
    image=Image.new("RGB",(1650,175+375*len(panels)),"white")
    draw=ImageDraw.Draw(image)
    def font(size,bold=False):
        return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",size)
    colors={"NLR snapshots":"#111827",LABELS[MODEL_IDS[0]]:"#111827",
        LABELS[MODEL_IDS[1]]:"#3375b9",LABELS[MODEL_IDS[2]]:"#df8230",
        LABELS[MODEL_IDS[3]]:"#7659a5","F1 aggregate rate":"#3375b9",
        "F2 aggregate rate":"#df8230","S1 conditional curve":"#7659a5"}
    draw.text((120,23),title,fill="#173348",font=font(30,True))
    draw.text((120,69),subtitle,fill="#4c606d",font=font(20))
    for index,p in enumerate(panels):
        left,right=130,1550; top,bottom=183+index*375,420+index*375
        draw.text((left,top-70),p["title"],fill="#18374e",font=font(23,True))
        values=np.concatenate([np.asarray(s[2],float) for s in p["series"]])
        values=values[np.isfinite(values)]
        lo,hi=float(values.min()),float(values.max()); pad=max((hi-lo)*.08,.01)
        ymin,ymax=(0 if p.get("zero") else lo-pad),hi+pad
        all_zero=bool(np.all(values==0))
        if all_zero: ymin,ymax=0.,5.
        xmin,xmax=p.get("xlim",(min(np.min(s[1]) for s in p["series"]),max(np.max(s[1]) for s in p["series"])))
        for y in np.linspace(ymin,ymax,6):
            py=bottom-(y-ymin)/(ymax-ymin)*(bottom-top)
            draw.line((left,py,right,py),fill="#dfe6eb")
            label=f"{y:,.0f}" if ymax>200 or all_zero else f"{y:.1f}"
            draw.text((left-15,py),label,anchor="rm",fill="#415562",font=font(18))
        for x in np.linspace(xmin,xmax,7):
            px=left+(x-xmin)/(xmax-xmin or 1)*(right-left)
            draw.line((px,top,px,bottom),fill="#ecf0f3")
            draw.text((px,bottom+12),f"{x:.0f}",anchor="mt",fill="#415562",font=font(18))
        for j,(label,x,y) in enumerate(p["series"]):
            color=colors[label]; lx=left+j*1420/max(3,len(p["series"]))
            draw.line((lx,top-29,lx+25,top-29),fill=color,width=3)
            draw.text((lx+34,top-40),label,fill="#304653",font=font(18))
            points=[(left+(float(xv)-xmin)/(xmax-xmin or 1)*(right-left),
                     bottom-(float(yv)-ymin)/(ymax-ymin)*(bottom-top)) for xv,yv in zip(x,y)]
            if len(points)>1: draw.line(points,fill=color,width=2)
        if all_zero:
            draw.text((left+30,top+70),"All observed and predicted Waiting = 0",fill="#4c606d",font=font(22))
        draw.line((left,top,left,bottom,right,bottom),fill="#5b6d79",width=2)
        draw.text((left,top-2),p["unit"],anchor="lb",fill="#415562",font=font(17))
        draw.text(((left+right)/2,bottom+42),p["xlabel"],anchor="mt",fill="#415562",font=font(18))
    draw.text((120,image.height-36),footer,fill="#4c606d",font=font(17))
    image.save(path)


def make_figures(out,segments,params,scores):
    example_records=[]
    for service in SERVICES:
        panels=[]
        for rate in (40.,90.):
            segment=next(s for s in segments.values() if s["service"]==service and s["rate"]==rate)
            obs=segment["observations"]
            times=np.arange(0,180.01,.5)
            prediction={m:simulate(m,params[service],rate,times,int(obs.running_requests.iloc[0]),
                int(obs.waiting_requests.iloc[0])) for m in MODEL_IDS[1:]}
            for state,column in (("running","running_requests"),("waiting","waiting_requests")):
                series=[("NLR snapshots",obs.replay_s.to_numpy(),obs[column].to_numpy())]
                for model in MODEL_IDS[1:]:
                    series.append((LABELS[model],times,prediction[model][state].to_numpy()))
                panels.append({"title":f"{rate:.0f} req/s: {state.capitalize()} requests",
                    "series":series,"xlabel":"Time from conditioned initial snapshot (s)",
                    "unit":"Requests","zero":True,"xlim":(0,180)})
            for model,pred in prediction.items():
                example_records.append({"service":service,"rate":rate,"model":model,
                    "initial_snapshot_relative_run_s":float(obs.relative_run_s.iloc[0]),
                    "trace":pred.to_dict("records")})
        plot(out/f"state_replay_{SHORT[service]}.png",f"{SHORT[service]}: conditional Running / Waiting replay",
            "Even-rate comparison. Initial state observed near run +60 s; subsequent states are model outputs.",panels,
            "Black: ~10 s NLR snapshots, joined for visibility. Models: event-driven; display grid 0.5 s. No power model.")
    panels=[]
    for service in SERVICES:
        for state in ("running","waiting"):
            series=[]
            for model in MODEL_IDS:
                s=scores[(scores.service==service)&(scores.model==model)].sort_values("rate")
                series.append((LABELS[model],s.rate.to_numpy(),s[f"{state}_rmse"].to_numpy()))
            panels.append({"title":f"{SHORT[service]}: {state.capitalize()} RMSE", "series":series,
                "xlabel":"Held-out configured rate (req/s)","unit":"Requests","zero":True})
    plot(out/"state_errors_by_rate.png","Candidate state replay errors by request rate",
        "Point errors at observed timestamps; first supplied initial state excluded. Lower is better.",panels,
        "B0 is a persistence reference, not a queue model. These are exploratory held-out rates, not an independent test.")
    panels=[]
    for service in SERVICES:
        p=params[service]; x=np.array(p["state_curve_running_knots"])
        panels.append({"title":SHORT[service],"series":[("F1 aggregate rate",x,x/p["short_lifetime_s"]),
            ("F2 aggregate rate",x,x/p["capacity_matched_lifetime_s"]),
            ("S1 conditional curve",x,p["state_curve_rate_knots"])],
            "xlabel":"Running requests","unit":"Normalized work / s","zero":True})
    plot(out/"fitted_service_curves.png","Different assumptions about concurrent service",
        "S1 empirical curve fitted only to odd-rate 60-150 s windows. Unit job work is a modeling assumption.",panels,
        "The service-rate curve is not GPU power, measured token throughput, or a confirmed SLO capacity.")
    dump(out/"illustrative_180s_traces.json",example_records)


def write_report(out,params,bands,overall,sensitivity,verification):
    def table(frame):
        f=frame.copy()
        for col in ("service",):
            if col in f: f[col]=f[col].replace(SHORT)
        if "model" in f: f["model"]=f["model"].replace(LABELS)
        return f.to_html(index=False,border=0,float_format=lambda v:f"{v:,.2f}")
    totals=overall[["service","model","running_rmse","waiting_rmse","unfinished_rmse"]]
    grouped=bands[["service","band","model","run_count","running_rmse","waiting_rmse"]]
    pars=pd.DataFrame([{"service":s,"K hypothesis":p["concurrency_limit_hypothesis"],
        "F1 lifetime (s)":p["short_lifetime_s"],"high-rate proxy (req/s)":p["conditional_high_load_rate_rps"],
        "F2 lifetime (s)":p["capacity_matched_lifetime_s"]} for s,p in params.items()])
    focus=bands[bands.model.eq("S1_state_dependent_lps")][["service","band","running_rmse","waiting_rmse"]]
    html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>④-2 NLR 처리 상태 후보 이월 모델 비교</title><style>
body{{font-family:'Malgun Gothic',Arial,sans-serif;max-width:1220px;margin:36px auto;padding:0 24px;color:#223447;line-height:1.8}}
h1{{font-size:28px}}h2{{font-size:22px;margin-top:34px}}table{{width:100%;border-collapse:collapse;font-size:13px;margin:16px 0}}
td,th{{padding:9px;border-bottom:1px solid #dbe3eb;text-align:right}}td:first-child,th:first-child{{text-align:left}}
th{{background:#eff3f7}}.eq{{font-family:Cambria,serif;font-size:20px;padding:13px;background:#f4f6f8}}
.note{{padding:16px 20px;background:#fff5df}}img{{max-width:100%;height:auto}}a{{color:#21659a}}code{{overflow-wrap:anywhere}}</style>
<h1>④-2 후보 이월 모델의 NLR Running·Waiting 재현 비교</h1>
<p>범위: NLR Llama-3.1-70B / TP=4 serving 단위의 조건부 상태 replay. IT 전력, 냉각, PCC, DFL은 결합하지 않았다.</p>
<p><strong>핵심 결론:</strong> 하나의 고정 처리시간은 저부하와 고부하 상태를 동시에 설명하기 어려웠다.
상태 의존 S1은 이번 비교의 전체 평균 오차를 줄였지만, 중간 요청률의 Running 증가와 Waiting 발생 시점은 아직 충분히 재현하지 못한다.
따라서 <strong>후속 검토 후보</strong>로 남기며, 전력 생성기에 바로 채택하지 않는다.</p>
<p class="note"><strong>결과 해석의 경계:</strong> 첫 Running·Waiting은 관측값을 넣고 이후 상태만 예측했다.
실제 요청별 유입 시각 대신 설정 요청률을 사용했다. 따라서 처음부터 요청을 입력해 전력까지 맞춘 검증이 아니다.
비교용 짝수 rate도 이전 연구에서 이미 살펴본 자료이며, 새로운 독립·blind test가 아니다.</p>
<h2>1. 비교한 모델과 출처</h2>
<table><tr><th>모델</th><th>구조</th><th>차용과 변경</th></tr>
<tr><td>B0</td><td>초기 Running·Waiting 유지</td><td>아무 변화도 예측하지 않는 진단용 기준선. 처리 보존모델 아님.</td></tr>
<tr><td>F1</td><td>고정 slot + 저부하 고정 처리시간</td><td>[43] FIFO·활성 구간 집계 구조*1를 단순화. token별 TTFT/TBT 대신 NLR 저부하 체류 proxy 사용.</td></tr>
<tr><td>F2</td><td>고정 slot + 처리용량에 맞춘 고정 처리시간</td><td>같은 구조에서 s=K/μ̃로 설정. 저부하와 고부하를 하나의 고정시간으로 설명할 수 있는지 비교.</td></tr>
<tr><td>S1</td><td>상태 의존 제한 처리공유(Sd-LPS)</td><td>Gupta & Zhang(2022)의 §2.1 식 (2)·(3)*2. NLR μ̂(R) 곡선 보정·단위 작업량 가정은 우리 변경사항.</td></tr></table>
<p><strong>F1/F2는 [43]의 전체 모델 재현이 아니다.</strong> 처리시간 분포의 변동, 입력/출력 token별 TTFT·TBT,
원문의 GMM·BiGRU 전력 모델은 구현하지 않았다. F1/F2가 맞지 않는다고 [43]의 원 모델을 반증했다고 해석하지 않는다.</p>
<h2>2. 모델 수식</h2>
<p class="eq">F1/F2: t<sub>i</sub><sup>start</sup> = max(t<sub>i</sub><sup>arrival</sup>, 첫 가용 slot 시각),
 t<sub>i</sub><sup>end</sup> = t<sub>i</sub><sup>start</sup> + s<br>
R(t) = Σ<sub>i</sub> 1{{t<sub>i</sub><sup>start</sup> ≤ t &lt; t<sub>i</sub><sup>end</sup>}} &nbsp; *1</p>
<p>F1의 s는 홀수 11–29 req/s 중 저변동 조건의 평균 Running/설정률 중앙값이다.
F2는 고부하의 조건부 이탈률 μ̃에 맞추기 위해 s=K/μ̃로 둔다. 모두 직접 측정한 request별 서비스시간은 아니다.</p>
<p class="eq">S1: R(t)=min(B(t),K), &nbsp; Q(t)=max(B(t)−K,0)<br>
처리 중 요청별 작업 진행률: ψ(R)=μ̂(R)/R &nbsp; (R&gt;0)<br>
요청 i 완료: ∫<sub>tᵢˢᵗᵃʳᵗ</sub><sup>t</sup> ψ(R(u)) du ≥ v<sub>i</sub> &nbsp; *2</p>
<p>S1은 원문의 처리공유 구조를 유지하고 모든 신규 요청에 정규화 작업량 vᵢ=1을 가정했다.
실측 token 수를 작업량 1로 측정했다는 뜻이 아니다. 요청별 잔여 작업량도 추적하므로 도착률이 줄어도 즉시 완료되지 않는다.</p>
<p>μ̂(R)는 홀수 rate의 60–150초 창에서 (평균 Running, 설정률−미완료 증가율) 쌍으로 구성한다.
60 req/s 미만 점들과 지속 대기 고부하 중앙값 anchor를 사용하며, 단조 회귀(PAVA)와 선형 보간을 적용했다.
단조 제약·보간·조건부 이탈률을 작업 처리율로 해석하는 것은 <strong>우리의 경험적 보정 가정</strong>이며 [43]이나 *2의 NLR 공식이 아니다.
평균 작업량을 1로 두고, 분석 창의 잔여 작업량 경계효과를 무시하는 근사가 들어간다.</p>
<h2>3. 보정·비교 자료 분리</h2>
<ul><li>보정: 홀수 rate 1–99 × 2 dataset = 100 run, 각 run의 60–150초.</li>
<li>비교: 짝수 rate 2–100 × 2 dataset = 100 run, 각 run의 60–240초 내 logger snapshot.</li>
<li>두 dataset의 같은 rate는 같은 split. ④-1의 전체 자료 추정치는 이번 모델 파라미터로 그대로 가져오지 않고 보정 자료에서 다시 계산했다.</li>
<li>첫 snapshot의 R·Q만 제공. 이후 관측값으로 재설정·보정하지 않으며, 첫 점은 오차 평가에서 제외.</li>
<li>초기 Running 요청의 잔여 작업량은 (0,1)의 균등 분위수로 배치. 제곱/제곱근 변형을 별도 민감도로 비교. 정확한 잔여량은 관측되지 않음.</li>
<li>요청 도착은 설정률의 규칙적 도착으로 근사. NLR burstiness=100000에 대응한 단순화이며 실측 arrival replay가 아님.</li>
<li>종료·정리 경계는 제외. 취소·재시도·거절 없이 요청 수를 보존한다는 후보모델 가정.</li></ul>
{table(pars)}<p>K는 보정 구간 Running의 최댓값을 사용한 가설이다. 장비·vLLM 설정값으로 확인한 한도가 아니다.</p>
<h2>4. 비교 결과</h2>
<p>표의 값은 run별 RMSE를 구한 뒤 동일 가중 평균한 값이다. 단위는 요청 개수. Running·Waiting을 임의 가중 J로 합치지 않았다.
이번에는 주어진 초기 상태와 입력에 대한 상태 궤적을 비교하므로 RMSE·MAE를 진단 지표로 쓴다.
확률적 residual의 서로 다른 표본 경로를 일대일 비교하는 이전 문제와 구분된다.</p>
{table(totals)}<img src="state_errors_by_rate.png" alt="요청률별 상태 재현 오차">
<h3>숫자를 어떻게 해석해야 하는가?</h3>
<ul><li><strong>F1:</strong> 저부하에서 추정한 약 3.1–3.2초를 모든 동시 처리 상태에 적용하면 고부하 요청을 너무 빨리 완료시킨다.
그래서 실제로 증가하는 Waiting이 모델에서는 사라진다.</li>
<li><strong>F2:</strong> 고부하 처리율에 맞춘 약 21–24초를 모든 상태에 적용하면 저부하에서도 Running을 지나치게 많이 유지한다.
고부하 Waiting은 잘 맞지만 저부하 Running 오차는 크다.</li>
<li><strong>S1:</strong> 동시 처리량에 따라 요청별 진행 속도를 바꾸어 두 고정시간 가정의 간극을 줄인다.
전체 Running RMSE는 F2 대비 두 서비스 모두 약 90% 감소했다. 그러나 고부하 Waiting 오차는 F2와 거의 같고,
InstructCoder 고부하에서는 F2가 근소하게 더 작다. 모든 지표·조건에서 S1이 우월하다는 뜻이 아니다.</li>
<li><strong>복잡도와 검증 한계:</strong> S1은 경험적 보간 곡선이 있어 고정시간 모델보다 유연하다.
보정에 사용하지 않은 rate에서 비교했지만, 복잡도를 동일하게 맞춘 비교나 외부 일반화 검증은 아니다.
절대적인 허용오차 기준도 사전 정의하지 않았으므로 이 결과만으로 최종 합격 판정을 내리지 않는다.</li></ul>
<h3>운전 영역별 결과</h3><p>low=1–30, transition=31–59, high=60–100 req/s. 설명용 구간이며 SLO/안정성 경계가 아니다.</p>
{table(grouped)}
<h3>S1의 남은 문제</h3>{table(focus)}
<ul><li>InstructCoder 40 req/s에서 Running RMSE는 약 156개이고, 마지막 비교 시점에는 관측보다 259개 많다.
180초 예시에서 보라색 선이 검은 선보다 가파르게 올라가는 부분이다.</li>
<li>Llama2-data 46·48 req/s에서 관측 Waiting의 최댓값은 각각 5·166개지만, S1은 두 경우 모두 Waiting=0으로 예측했다.
대기 발생 여부·시점을 정밀하게 판정하는 모델로는 아직 부족하다.</li>
<li>고부하에서는 F2와 S1이 같은 조건부 처리율을 사용하므로 두 선이 거의 겹친다.
따라서 그 구간에서 S1만의 동특성을 확인한 것은 아니다.</li>
<li>초기 상태를 계속 유지하는 B0도 고부하 Running은 잘 맞을 수 있다. 해당 상태가 이미 한도 근처에 있기 때문이다.
Waiting의 증가와 함께 보아야 미완료 작업 이월이 제대로 표현되는지 판단할 수 있다.</li></ul>
<h2>5. 180초 예시</h2>
<p>가로축 0은 run 시작이 아니라 약 60초 이후의 첫 관측 상태를 넣은 시점이다. 검은 선은 약 10초 간격 snapshot을 연결한 것이다.
0.5초 모델 출력선과 비교하되 실측이 0.5초라는 뜻은 아니다. 마지막 logger 점 이후의 짧은 모델 구간은 오차 계산에 사용하지 않았다.</p>
<img src="state_replay_InstructCoder.png" alt="InstructCoder Running Waiting 비교">
<img src="state_replay_Llama2-data.png" alt="Llama2-data Running Waiting 비교">
<h2>6. 초기 잔여량·도착 위상 민감도</h2>
<p>각 서비스 20·40·60·90 req/s에서 S1의 초기 잔여 작업 분위수 u를 u², u, √u로 변경하고,
도착 위상을 한 요청 간격의 0.1/0.5/0.9로 변경했다. 모델 재보정 없이 총 72개 경우를 비교했다. 아래는 각 조건의 오차 범위다.</p>
{table(sensitivity)}
<p>초기 잔여 작업량과 실제 arrival 부재 때문에 같은 R·Q로 시작해도 경로가 유일하게 정해지지 않는다.
이 민감도는 신뢰구간이 아니며, token별 작업량 분포의 불확실성을 모두 포괄하지도 않는다.</p>
<h2>7. 이번 결과로 말할 수 있는 범위</h2>
<ul><li>고정 처리시간 후보와 상태 의존 후보가 저부하·누적 구간·과부하를 동시에 설명하는 정도를 비교했다.</li>
<li>실제 μ(R)만으로 충분한지, token 조성·cache·배치 정책·시간에 따른 성능 변화가 추가로 필요한지는 남은 문제다.</li>
<li>정상상태 조건 사이의 상태 재현이 좋아도, 시간가변 arrival 전이응답 검증이나 요청별 latency/SLO 검증이 완료된 것은 아니다.</li>
<li>5분 평균 전력 또는 시설 에너지의 오차는 평가하지 않았다. M3 입력을 완료율이나 Running으로 바꾸지 않았다.</li>
<li><strong>자동 최종 모델 선정은 하지 않았다.</strong> 모형 간 상대 성능과 남은 오차를 보고 후속 보정/전력 연결 여부를 판단해야 한다.</li></ul>
<p><strong>권장 다음 단계:</strong> 중간 요청률의 오차가 초기 잔여 작업량, 실제 admission과 설정률의 차이,
요청별 작업량 차이 또는 시간에 따른 처리 성능 변화 중 무엇과 관련되는지 먼저 점검한다.
새 후보가 필요하면 이번 비교 세트의 지위를 유지하지 말고 후속 탐색으로 명시한다.
그다음 시간가변 요청 입력과 감소 후 배출(drain) 자료의 확보 가능성을 확인한 뒤 전력 연결을 검토한다.</p>
<h2>8. 수치·재현성 확인</h2>
<p>모든 기계적 후보의 관측 시점에서 B=B₀+A−C 확인. 미래 도착 선처리·음수 상태·구간 경계 초기화 방지 테스트 포함.
이벤트 기반 계산이므로 0.1/0.5/1초는 내부 적분 간격이 아니라 출력 간격이다.
공통 시각 출력 일치 확인 결과: {verification['sampling_comparison_max_count_difference']} 요청 차이.
기존 파일 {verification['protected_file_count']}개 해시 유지.</p>
<p><a href="protocol.json">사전 비교 규칙</a> · <a href="fitted_parameters.json">보정 파라미터</a> ·
<a href="per_run_scores.json">run별 전체 지표</a> · <a href="verification.json">검증 결과</a> ·
<a href="source_manifest.json">입력 출처·해시</a></p>
<h2>Reference와 차용 범위</h2>
<p>*1 [43] G. Wilkins, F. Kazhamiaka, R. Rajagopal, <a href="https://arxiv.org/html/2603.18383v1#S3.SS3">From Servers to Sites</a>, arXiv:2603.18383v1, 2026, §3.3, Eq. (6).
활성 구간 집계·FIFO 구조 참고. 고정 수명으로의 축소와 NLR 보정은 본 연구의 변경.</p>
<p>*2 V. Gupta and J. Zhang, <a href="https://doi.org/10.1287/stsy.2021.0087">Approximations and Optimal Control for State-Dependent Limited Processor Sharing Queues</a>, Stochastic Systems, 12(2), 205–225, 2022, §2.1, Eqs. (2)–(3).
처리공유와 동시 처리 한도 구조 차용. 원문의 확산근사·최적 동시성 제어는 사용하지 않았다. NLR μ̂(R) 보정 방식은 원문에 없음.</p>
<p>*3 NLR [3] online-rate 원자료 및 기존 ③의 logger 추출. <a href="https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/metrics/loggers.py">vLLM v0.11.0 logger</a>의 Running·Waiting은 scheduler snapshot.
실제 admission/completion 시각과 서비스 잔여량은 확보되지 않았다.</p></html>'''
    (out/"report_ko.html").write_text(html,encoding="utf-8")


def audit_saved_results(out):
    """Independently recompute scalar metrics, conservation and raw snapshot links."""
    read=lambda name:json.loads((out/name).read_text(encoding="utf-8"))
    scores={(r["run_id"],r["model"]):r for r in read("per_run_scores.json")}
    raw=pd.read_csv(ROOT/"outputs/nlr_processing_diagnostic_v1/raw_server_metrics.csv").set_index("source_line")
    max_difference=0.; count=0; snapshots=0
    for r in read("observed_time_replays.json"):
        p=pd.DataFrame(r["prediction"]); key=(r["run_id"],r["model"])
        original=raw.loc[r["source_lines"]]
        np.testing.assert_array_equal(original.running_requests,r["observed_running"])
        np.testing.assert_array_equal(original.waiting_requests,r["observed_waiting"])
        for state in ("running","waiting","unfinished"):
            actual=np.asarray(r["observed_running"],float) if state=="running" else np.asarray(r["observed_waiting"],float)
            if state=="unfinished": actual=np.asarray(r["observed_running"],float)+np.asarray(r["observed_waiting"],float)
            error=p[state].to_numpy()[1:]-actual[1:]
            value=float(np.sqrt(np.mean(error**2)))
            max_difference=max(max_difference,abs(value-scores[key][f"{state}_rmse"]))
        if r["model"]!="B0_persistence":
            np.testing.assert_array_equal(p.unfinished,p.unfinished.iloc[0]+p.arrivals-p.completions)
            assert (p.running>=0).all() and (p.waiting>=0).all()
            assert (p.arrivals.diff().dropna()>=0).all() and (p.completions.diff().dropna()>=0).all()
        count+=1; snapshots+=len(p)
    assert count==400 and max_difference<1e-10
    return {"independent_saved_RMSE_recalculation_max_difference":max_difference,
        "saved_trajectory_count":count,"raw_linked_snapshots_including_repeated_models":snapshots,
        "raw_snapshot_values_and_saved_conservation_verified":True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/carryover_replay_v1")
    parser.add_argument("--render-existing",action="store_true",help="Re-audit/re-render this stage's existing outputs; never refit or change scored trajectories")
    args=parser.parse_args(); out=args.output.resolve()
    if args.render_existing:
        read=lambda name:json.loads((out/name).read_text(encoding="utf-8"))
        assert read("protocol.json")["stage"]=="4-2: conditional NLR Running/Waiting replay"
        manifest=read("source_manifest.json")
        for item in manifest:
            if item["role"]!="builder": assert sha256(Path(item["path"]))==item["sha256"]
        verification=read("verification.json"); verification.update(audit_saved_results(out))
        params=read("fitted_parameters.json"); scores=pd.DataFrame(read("per_run_scores.json"))
        bands,overall=group_scores(scores)
        sens=pd.DataFrame(read("initial_state_sensitivity.json"))
        sensitivity=sens.groupby(["service","rate"],as_index=False).agg(
            running_RMSE_min=("running_rmse","min"),running_RMSE_max=("running_rmse","max"),
            waiting_RMSE_min=("waiting_rmse","min"),waiting_RMSE_max=("waiting_rmse","max"))
        segments=make_segments(pd.read_csv(ROOT/"outputs/nlr_transition_audit_v1/inference_online_rate_llama3_70b_runs.csv",dtype={"id":str}),
            pd.read_csv(ROOT/"outputs/nlr_processing_diagnostic_v1/raw_server_metrics.csv"))
        make_figures(out,segments,params,scores)
        write_report(out,params,bands,overall,sensitivity,verification)
        dump(out/"verification.json",verification)
        for item in manifest:
            if item["role"]=="builder": item["sha256"]=sha256(Path(__file__))
        dump(out/"source_manifest.json",manifest)
        print(json.dumps(verification,indent=2)); return
    if out.exists(): raise FileExistsError(f"Refusing overwrite: {out}")
    sources={"runs":ROOT/"outputs/nlr_transition_audit_v1/inference_online_rate_llama3_70b_runs.csv",
        "raw_metrics":ROOT/"outputs/nlr_processing_diagnostic_v1/raw_server_metrics.csv",
        "protocol":ROOT/"configs/carryover_replay_protocol_v1.json"}
    protected=list((ROOT/"src/nlr_data").glob("*.py"))
    for folder in ("carryover_parameter_audit_v1","nlr_processing_diagnostic_v1","facility_arrival_scaling_v1"):
        protected.extend(p for p in (ROOT/"outputs"/folder).glob("*") if p.is_file())
    before={str(p):sha256(p) for p in protected}
    protocol=json.loads(sources["protocol"].read_text(encoding="utf-8"))
    runs=pd.read_csv(sources["runs"],dtype={"id":str})
    raw=pd.read_csv(sources["raw_metrics"])
    segments=make_segments(runs,raw)
    params,calibration=fit_parameters(segments)
    train_ids={s["run_id"] for s in segments.values() if s["split"]=="calibration"}
    compare_ids=set(segments)-train_ids
    assert len(train_ids)==len(compare_ids)==100 and not train_ids&compare_ids
    # Independence of numerical fitting from comparison data, not proof of a blind study.
    only_train={k:s for k,s in segments.items() if k in train_ids}
    assert fit_parameters(only_train)[0]==params
    out.mkdir(parents=True)
    dump(out/"protocol.json",protocol)
    dump(out/"fitted_parameters.json",params)
    dump(out/"calibration_evidence.json",calibration)
    scores=[]; trajectories=[]; initial_limit_violations=[]; conservation_max=0
    for index,segment in enumerate(s for s in segments.values() if s["split"]=="comparison"):
        obs=segment["observations"]; service=segment["service"]; rate=segment["rate"]
        r0,q0=int(obs.running_requests.iloc[0]),int(obs.waiting_requests.iloc[0])
        if r0>params[service]["concurrency_limit_hypothesis"]:
            initial_limit_violations.append(segment["run_id"])
            raise ValueError("Held-out initial state exceeds K hypothesis; do not silently clip")
        for model in MODEL_IDS:
            pred=simulate(model,params[service],rate,obs.replay_s.to_numpy(),r0,q0)
            stats=score(obs,pred)
            scores.append({"run_id":segment["run_id"],"service":service,"rate":rate,"model":model,
                "band":"low" if rate<=30 else "transition" if rate<60 else "high",**stats})
            trajectories.append({"run_id":segment["run_id"],"service":service,"rate":rate,"model":model,
                "observed_running":obs.running_requests.tolist(),"observed_waiting":obs.waiting_requests.tolist(),
                "source_lines":obs.source_line.tolist(),"prediction":pred.to_dict("records")})
            if model!="B0_persistence": conservation_max=max(conservation_max,int(abs(pred.conservation_error).max()))
        if (index+1)%25==0: print(f"Compared {index+1}/100 held-out runs",flush=True)
    scores=pd.DataFrame(scores)
    bands,overall=group_scores(scores)
    dump(out/"per_run_scores.json",scores.to_dict("records"))
    dump(out/"grouped_scores.json",{"overall":overall.to_dict("records"),"by_band":bands.to_dict("records")})
    dump(out/"observed_time_replays.json",trajectories)
    sens=[]; max_sampling_difference=0
    selected=[s for s in segments.values() if s["rate"] in (20.,40.,60.,90.)]
    for segment in selected:
        service,rate,obs=segment["service"],segment["rate"],segment["observations"]
        r0,q0=int(obs.running_requests.iloc[0]),int(obs.waiting_requests.iloc[0])
        for phase in (.1,.5,.9):
            for power in (.5,1.,2.):
                pred=simulate("S1_state_dependent_lps",params[service],rate,obs.replay_s.to_numpy(),r0,q0,
                              phase=phase,work_power=power)
                sens.append({"service":service,"rate":rate,"arrival_phase":phase,"initial_work_power":power,**score(obs,pred)})
        outputs={dt:simulate("S1_state_dependent_lps",params[service],rate,np.arange(0,180+.01,dt),r0,q0)
                 for dt in (.1,.5,1.)}
        one=outputs[1.][["running","waiting","unfinished"]].to_numpy()
        for dt,stride in ((.1,10),(.5,2)):
            delta=np.max(abs(outputs[dt][["running","waiting","unfinished"]].to_numpy()[::stride]-one))
            max_sampling_difference=max(max_sampling_difference,int(delta))
    sens=pd.DataFrame(sens)
    sensitivity=sens.groupby(["service","rate"],as_index=False).agg(
        running_RMSE_min=("running_rmse","min"),running_RMSE_max=("running_rmse","max"),
        waiting_RMSE_min=("waiting_rmse","min"),waiting_RMSE_max=("waiting_rmse","max"))
    dump(out/"initial_state_sensitivity.json",sens.to_dict("records"))
    make_figures(out,segments,params,scores)
    print("Figures generated; checking source preservation",flush=True)
    assert all(sha256(Path(p))==h for p,h in before.items())
    verification={"created_utc":datetime.now(timezone.utc).isoformat(),"all_checks_passed":True,
        "calibration_runs":len(train_ids),"comparison_runs":len(compare_ids),"candidate_score_rows":len(scores),
        "training_parameters_unchanged_after_comparison_removed":True,"conservation_max_error":conservation_max,
        "sampling_comparison_max_count_difference":max_sampling_difference,
        "sensitivity_cases":len(sens),"protected_file_count":len(before),"protected_files_unchanged":True,
        "power_model_modified":False,"independent_test_claim":False,"automatic_final_model_selection":False}
    if conservation_max!=0 or max_sampling_difference!=0: raise AssertionError("Conservation or output-grid agreement failed")
    verification.update(audit_saved_results(out))
    dump(out/"verification.json",verification)
    dump(out/"source_manifest.json",[{"role":role,"path":str(p),"sha256":sha256(p)} for role,p in sources.items()]
        +[{"role":"builder","path":str(Path(__file__)),"sha256":sha256(Path(__file__))},
          {"role":"model_code","path":str(ROOT/"src/aidc_facility/carryover_replay.py"),"sha256":sha256(ROOT/"src/aidc_facility/carryover_replay.py")}])
    write_report(out,params,bands,overall,sensitivity,verification)
    print(overall[["service","model","running_rmse","waiting_rmse","unfinished_rmse"]].to_string(index=False))
    print(json.dumps(verification,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
