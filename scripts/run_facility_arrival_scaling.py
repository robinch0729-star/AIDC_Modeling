"""Compare matched-count facility arrival constructions on the complete common trace.

All generated counts are offered arrivals, not admitted or completed requests.
This script does not import/run M3/R3, route to instances, or simulate queues.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aidc_facility.facility_arrivals import (SERVICES, METHODS, aggregate_counts,
    scale_arrivals, arrival_metrics, relative_within_block_rms, build_source_seconds, sha256)

LABELS = {"replay": "Observed 1-s shape x K", "conditional_60s": "1-min totals + uniform arrivals",
          "conditional_300s": "5-min totals + uniform arrivals"}
COLORS = {"replay": "#ba3b34", "conditional_60s": "#276ba8", "conditional_300s": "#21826a",
          "Code": "#276ba8", "Conversation": "#bc671a", "Total": "#303b49"}


def chart(path, title, subtitle, panels, footer):
    """Static scientific plot, not an AI-generated graphic."""
    from PIL import Image, ImageDraw, ImageFont
    width, height = 1650, 180 + 390*len(panels)
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    def f(size, bold=False):
        return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", size)
    d.text((105, 24), title, font=f(31, True), fill="#152c40")
    d.text((105, 69), subtitle, font=f(20), fill="#4b5966")
    for i, panel in enumerate(panels):
        left, right = 130, 1555
        top, bottom = 178+i*390, 431+i*390
        d.text((left, top-68), panel["title"], font=f(24, True), fill="#243e52")
        lx = left
        for name in panel["series"]:
            label = LABELS.get(name, name)
            color = COLORS[name]
            d.line((lx, top-26, lx+28, top-26), fill=color, width=3)
            d.text((lx+36, top-38), label, font=f(18), fill="#293d50")
            lx += 465
        vals = np.concatenate([np.asarray(y) for y in panel["series"].values()])
        low, high = float(vals.min()), float(vals.max())
        if panel.get("zero", False):
            low = 0.
        pad = max((high-low)*.07, high*.01, .001)
        ymin, ymax = (max(0., low-pad) if low >= 0 else low-pad), high+pad
        if panel.get("zero", False):
            ymin = 0.
        x = np.asarray(panel["x"], float)
        xmin, xmax = float(x.min()), float(x.max())
        xx = left + (x-xmin)/(xmax-xmin or 1)*(right-left)
        for tick in np.linspace(ymin, ymax, 6):
            yy = bottom-(tick-ymin)/(ymax-ymin)*(bottom-top)
            d.line((left, yy, right, yy), fill="#e1e6eb")
            label = f"{tick:,.0f}" if ymax>100 else f"{tick:.2f}"
            d.text((left-16, yy), label, anchor="rm", font=f(18), fill="#364757")
        for idx in np.unique(np.linspace(0, len(x)-1, 7).astype(int)):
            px = xx[idx]
            d.line((px, top, px, bottom), fill="#edf0f4")
            label = panel.get("ticks", [])[idx] if "ticks" in panel else f"{x[idx]:g}"
            d.text((px, bottom+17), label, anchor="mt", font=f(18), fill="#364757")
        for name, y in panel["series"].items():
            yy = bottom-(np.asarray(y)-ymin)/(ymax-ymin)*(bottom-top)
            d.line(list(zip(xx, yy)), fill=COLORS[name], width=2)
        d.line((left, top, left, bottom, right, bottom), fill="#526575", width=2)
        d.text((left, top-1), panel["unit"], anchor="lb", font=f(18), fill="#364757")
        d.text(((left+right)//2, bottom+44), panel["xlabel"], anchor="mt", font=f(19), fill="#364757")
    d.text((105, height-46), footer, font=f(19), fill="#4b5966")
    img.save(path)


def write_report(output, summary, metrics, variability, source_diag, selected):
    k = summary["primary_multiplier"]
    seed = summary["seeds"][0]
    table = metrics[(metrics.multiplier.eq(k)) & (metrics.phase_offset_s.eq(0)) &
                    (metrics.seed.eq(seed)) & metrics.service.eq("total") & metrics.bin_s.isin([1, 300])]
    table = table[["method", "bin_s", "mean_rps", "peak_to_mean", "adjacent_change_p95_pct_mean"]].copy()
    table.columns = ["도착 방식", "집계 간격(s)", "평균 요청률(req/s)", "최대/평균", "인접 구간 변화 P95 / 평균(%)"]
    table.iloc[:, 2:] = table.iloc[:, 2:].round(4)
    mix = summary["service_mix"]
    source_table = pd.DataFrame([{"서비스": s, "원자료 요청 수": mix[s]["requests"],
        "전체 요청 구성비(%)": round(mix[s]["fraction"]*100, 4),
        "5분별 구성비 최소(%)": round(mix[s]["min_5min_fraction"]*100, 2),
        "5분별 구성비 최대(%)": round(mix[s]["max_5min_fraction"]*100, 2)} for s in SERVICES])
    scales = pd.DataFrame(summary["scale_cases"])
    diag = source_diag.round(4).to_html(index=False, border=0)
    document = f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>② 시설 규모 요청량 구성 비교</title>
<style>body{{font-family:"Malgun Gothic",Arial,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#1c3041;line-height:1.8}}h1{{font-size:29px}}h2{{font-size:23px;margin-top:32px}}h3{{font-size:19px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #d9e1e8;text-align:left}}th{{background:#eff3f7}}img{{width:100%;height:auto}}.note{{background:#f4f7fa;border-left:4px solid #466b89;padding:14px 20px}}.formula{{font-family:Cambria,serif;font-size:21px;padding:12px;background:#f7f8fa}}a{{color:#20649b}}code{{font-family:Consolas,monospace}}li{{margin:5px 0}}</style>
<h1>② 시설 규모 요청량 구성 비교</h1>
<p>Azure 서비스 요청열을 시설 규모의 <strong>도착 요청량</strong>으로 확대하는 방법을 구현·비교하였다. IT 전력, instance 배정, 처리능력, 대기열은 계산하지 않았다.</p>
<div class="note"><strong>결론:</strong> 동일한 5분 총 요청량도 구간 내부의 도착 가정에 따라 1초 피크가 크게 달라질 수 있다. 현재 자료만으로 가장 현실적인 단기 도착 모델을 확정할 수 없다. 따라서 5분 총량을 공통 입력으로 고정하고, 세 도착 가정을 ③·④의 민감도 입력으로 유지한다. 파형이 부드럽다는 이유로 모델을 선정하지 않는다.</div>
<h2>1. 완료 범위와 미확정 항목</h2><ul>
<li>공통 관측 구간: {summary['start_utc']} 이상, {summary['end_exclusive_utc']} 미만. 약 119시간 50분, 서비스별 1,438개 5분 구간.</li>
<li>원본 요청을 1초 단위로 재집계하고 기존 5분 감사 결과의 모든 구간과 일치 확인. 1초는 이번 도착량 진단의 해상도이며, 실제 전력 측정 간격이나 최종 내부 계산 간격을 확정한 것이 아님.</li>
<li>두 서비스에 같은 확대 배수 K 적용. 전체 구성비는 약 39:61이지만 매 순간 39:61로 강제하지 않음.</li>
<li>새 합성 요청열은 관측 trace와 분리 저장. M3·R3와 기존 IT 프로파일은 수정하지 않음.</li>
<li><strong>미확정:</strong> 최종 K, 시설 instance 수·배정 비율, 허용 요청률·SLO, 17.39 MW IT 평균 달성, queue와 전력 전이.</li></ul>
<h2>2. 요청 구성비와 규모</h2>{source_table.to_html(index=False,border=0)}
<p>*1: 요청 수·시각은 Azure 공개 자료에서 집계. 두 공개 표본의 조합이며 실제 한 시설의 서비스 구성이나 시장 평균이라는 뜻은 아니다. Code→InstructCoder, Conversation→Llama2-data 대응은 우리 시나리오의 서비스 매핑이며 token 분포·모델·하드웨어의 동등성을 뜻하지 않는다.</p>
<p class="formula">A<sup>facility</sup><sub>d,b</sub> = K · A<sup>Azure</sup><sub>d,b</sub>, &nbsp; b: 5분 구간</p>
<p>이 식은 요청 수를 확대하는 시나리오 정의이며 물리적으로 식별된 규모 법칙이 아니다. 같은 K를 써서 전체 구성비와 5분별 두 서비스의 상대 변화를 유지한다. 정수 K만 사용하여 이번 비교에서는 반올림 오차가 없다.</p>
{scales.round(4).to_html(index=False,border=0)}
<p>K=100·1000·3000은 결과의 규모 의존성을 보기 위한 예시다. K는 서버 수가 아니고 어떤 경우도 17.39 MW와 대응시킨 결과가 아니다. 전력 목표에 맞춰 도착열을 사후 보정하지 않았다.</p>
<img src="facility_requests_5min.png" alt="서비스별 5분 도착 요청률">
<h2>3. 같은 총량으로 비교한 세 가지 도착 가정</h2>
<table><tr><th>후보</th><th>유지하는 자료</th><th>새로 둔 가정과 한계</th></tr>
<tr><td>A. replay</td><td>관측 1초 요청 수를 그대로 K배</td><td>모든 짧은 변동까지 공통 신호처럼 확대. 원형 보존 비교안이며 모든 시설의 변동 상한은 아님.</td></tr>
<tr><td>B. conditional_60s</td><td>관측 1분별·5분별 총 요청 수</td><td>각 1분 내부의 요청 도착 시각을 독립 균등 생성. 1분 미만의 실제 burst·자기상관을 제거할 수 있음.</td></tr>
<tr><td>C. conditional_300s</td><td>관측 5분별 총 요청 수</td><td>각 5분 내부에서 균등 생성. 실제 1분 수요 변화까지 제거할 수 있음.</td></tr></table>
<p class="formula">(a<sub>1</sub>, …, a<sub>B</sub>) | n ~ Multinomial(n; 1/B, …, 1/B), &nbsp; n = K·A<sup>Azure</sup><sub>d,parent</sub></p>
<p>*3: B는 60 또는 300개의 1초 구간. 고정된 구간 내 도착 건수에 조건을 건 Poisson 과정의 균등 도착 성질을 이용한다. 구간 총량을 고정했으므로 각 초의 건수는 서로 독립 Poisson이 아니라 조건부 multinomial이다. Azure 자체가 Poisson이라는 검정 결과는 아니다. 60·300초 선택과 시설 확대에의 적용은 우리 연구의 가정이다.</p>
<p>재생성은 같은 parent 구간 내부의 도착 시각을 바꾸는 <strong>오프라인 합성</strong>이다. 관측된 실제 도착 시각을 보존한 결과가 아니며, 실시간 예측 입력에 미래 구간 총량을 제공해서는 안 된다. 서로 다른 parent 구간 간 요청을 이동하지 않는다. 생성된 초별 요청에는 개별 token mark를 배정하지 않았다.</p>
<h2>4. 비교 결과</h2>
<p>아래는 K={k}, 첫 seed={seed}의 전체 시설 도착 요청량이다. 세 seed 전체 결과도 저장했다. ‘인접 구간 변화’는 요청률 변화이며 MW/s의 전력 ramp가 아니다.</p>
{table.to_html(index=False,border=0)}
<img src="facility_requests_180s_comparison.png" alt="동일 구간 180초 도착 요청률 비교">
<p>확대 구간 선택: {summary['zoom_start_utc']}부터 180초. 전체 관측 구간에서 Code의 관측 10초 요청률 변화 절댓값이 가장 큰 지점 주변을 사전 정의한 규칙으로 선택하였다. 대표적인 모든 구간을 뜻하지 않는다.</p>
<img src="facility_requests_scale_sensitivity.png" alt="규모 확대와 분내 상대 변동 민감도">
<p>분내 상대 변동은 각 초의 요청 수에서 해당 1분 평균을 뺀 값의 RMS를 기간 전체 평균으로 나눈 값이다. A는 K에 무관하게 상대 변동이 같고, B의 독립 도착 성분은 K가 커질수록 상대적으로 작아진다. 이는 생성 가정의 수학적 결과이지 B가 실제 시설에 더 적합하다는 증거는 아니다.</p>
<h3>원자료 단기 변동 진단</h3>{diag}
<p>표의 within_minute_dispersion은 Σ(a−분평균)² / Σa이다. 분내 일정 강도의 조건부 균등 도착 모형에서는 기대값이 59/60이다. 차이는 도착의 군집성 또는 분내 강도 변화 등에서 올 수 있으며, 하나의 수치만으로 원인을 구분하거나 노이즈라고 단정하지 않는다. 일별·구간 위상별 진단도 별도 CSV에 저장했다.</p>
<p>이번 관측값은 Code 약 1.47, Conversation 약 1.17로 위 단순 모형의 기대값 약 0.98과 차이가 있다. 따라서 원자료의 모든 초 단위 변동을 독립적인 표본 잡음이라고 간주하여 제거할 근거는 없다. B·C는 관측 단기 변동을 덜 유지하는 민감도 가정이지 원자료를 교정한 결과가 아니다.</p>
<h2>5. 이번 단계에서 채택한 절차</h2><ol>
<li><strong>5분 총 도착 요청량은 관측 trace의 실제 시간 순서를 유지</strong>하고, 같은 K로 서비스별 확대.</li>
<li>5분 내부의 실제 시설 동시성은 미식별이므로 A·B·C 모두 보관. 한 후보를 ‘현실 검증 완료’로 지정하지 않음.</li>
<li>다음 ③에서 NLR의 Running·Waiting·처리량을 분석하여 조건별 안정 운전범위를 확인.</li>
<li>그다음 전체 요청 규모와 고정 instance 배정을 함께 정하고, ④에서 도착 가정별 미완료 작업 이월 영향을 비교.</li>
<li>이 비교가 끝나기 전에는 요청 피크를 곧바로 GPU 과부하·queue·전력 ramp로 해석하지 않음.</li></ol>
<h2>6. 검증과 자료 경계</h2><ul>
<li>모든 확대 후보에서 서비스별 5분 총량 정확 보존. 같은 날짜·시각, 같은 분석 길이로 비교.</li>
<li>1·10·60·300초 집계, 10·60·300초의 반 구간 offset 비교. 위상 이동 시 경계의 불완전 구간은 제외하며 제외 길이·요청 수 명시.</li>
<li>공개 trace 파일 SHA-256 재확인, 기존 M3·R3·180초 표본·기존 시간가변 모델 결과의 hash 보존 확인.</li>
<li>미검증: 실제 facility 요청열, 서비스 간 표본의 대표성, 일주일 이상 일반화, request↔전력 전이, token별 처리량, instance간 분배.</li>
<li>1분·5분 경계의 강도 변화는 B·C에도 남을 수 있다. 전력 전이를 해결하거나 실제 0.1초 파형을 검증한 단계가 아니다.</li></ul>
<h2>7. 참고문헌과 차용 범위</h2><ul>
<li>*1 <a href="https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md">Azure LLM inference trace 2024</a>, CC BY 4.0; Stojkovic et al., <a href="https://doi.org/10.1109/HPCA61900.2025.00102">DynamoLLM, IEEE HPCA 2025</a>. 도착 시각·token 자료를 직접 사용. 39.17:60.83은 우리 공통 기간 재집계.</li>
<li>*2 Paper list [3], Vercellino et al., <a href="https://arxiv.org/html/2604.07345v1#S2.SS3.SSS2">Measurement of Generative AI Workload Power Profiles for Whole-Facility Data Center Infrastructure Planning</a>, 2026 preprint, §2.3.2·§4.2. 서비스 요청열→instance→node 전력의 구조 참고. 이번 multinomial 방식이나 60초 설정은 [3]의 그대로 재현이 아님.</li>
<li>*3 Larson &amp; Odoni, <a href="https://web.mit.edu/urban_or_book/www/book/chapter2/2.12.3.html">Urban Operations Research, §2.12.3 Unordered Arrival Times</a>, MIT. 고정 건수에 조건부인 균등 도착 원리만 차용. Azure 적합성이나 시설 규모 정당화 근거로 사용하지 않음.</li></ul>
<p>결과 파일: <a href="comparison_metrics.csv">비교 지표</a> · <a href="facility_demand_5min.csv">규모별 5분 도착량</a> · <a href="run_summary.json">설정·결과</a> · <a href="verification.json">실행 검증</a> · <a href="independent_verification.json">독립 재계산 검증</a></p>
</html>'''
    (output / "report_ko.html").write_text(document, encoding="utf-8")


def run(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    scales, seeds = cfg["comparison_multipliers"], cfg["seeds"]
    primary = cfg["primary_multiplier_for_figures"]
    if primary not in scales or len(set(scales)) != len(scales):
        raise ValueError("Invalid scale configuration")
    preserved = [ROOT / "outputs/r3_high_rate_benchmark/train_validation_m3_parameters.csv",
                 ROOT / "outputs/time_varying_it_v1/run_config.json",
                 ROOT / "data/raw/azure_inference_sample/code_arrival_sample.csv",
                 ROOT / "data/raw/azure_inference_sample/conversation_arrival_sample.csv",
                 ROOT / "outputs/azure_long_trace_audit_v1/audit_summary.json"]
    old_cfg = json.loads(preserved[1].read_text(encoding="utf-8"))
    preserved += [ROOT / "outputs/r3_high_rate_benchmark" / name for name in old_cfg["locked_parameters"]]
    before = {str(p): sha256(p) for p in set(preserved)}
    counts, provenance, five = build_source_seconds(args.raw_dir, args.audit_dir, output, args.chunksize)
    start = pd.Timestamp(provenance["start_utc"])
    n = len(counts["code"])
    checks = []
    def check(name, result, **values):
        checks.append({"check": name, "passed": bool(result), **values})
        if not result:
            raise AssertionError(name)
    overall = sum(int(v.sum()) for v in counts.values())
    common_five = {s: aggregate_counts(counts[s], 300) for s in SERVICES}
    total_five = sum(common_five.values())
    mix = {s: {"requests": int(counts[s].sum()), "fraction": float(counts[s].sum()/overall),
               "min_5min_fraction": float(np.min(common_five[s]/total_five)),
               "max_5min_fraction": float(np.max(common_five[s]/total_five))} for s in SERVICES}
    metrics, variation, daily_rows, diagnostics = [], [], [], []
    saved = {}
    dates = pd.date_range(start, periods=n, freq="s").strftime("%Y-%m-%d").to_numpy()
    day_indices = {d: np.flatnonzero(dates == d) for d in np.unique(dates)}
    for service in SERVICES:
        x = counts[service].reshape(-1, 60).astype(float)
        dispersion = float(((x-x.mean(axis=1, keepdims=True))**2).sum()/x.sum())
        diagnostics.append({"service": service, "within_minute_dispersion": dispersion,
                            "uniform_conditional_expected": 59/60,
                            "within_minute_relative_rms_pct": relative_within_block_rms(counts[service])*100})
        for dt in cfg["analysis_bin_seconds"]:
            metrics.append({"method": "source_observed", "multiplier": 1, "seed": -1,
                            "service": service, **arrival_metrics(counts[service], dt)})
    demand_frames = []
    for k in scales:
        demand = five.copy()
        demand["multiplier"] = k
        demand["facility_offered_requests"] = demand.request_count*k
        demand["facility_offered_rps"] = demand.facility_offered_requests/300
        demand["scaled_input_tokens_block_total"] = demand.ContextTokens_sum_valid*k
        demand["scaled_output_tokens_block_total"] = demand.GeneratedTokens_sum_valid*k
        demand["token_timing_assigned"] = False
        demand_frames.append(demand)
        for method in METHODS:
            for seed in seeds[:1] if method == "replay" else seeds:
                generated = {s: scale_arrivals(counts[s], k, method,
                    np.random.SeedSequence([seed, i, k])) for i, s in enumerate(SERVICES)}
                for s in SERVICES:
                    target = common_five[s]*k
                    actual = aggregate_counts(generated[s], 300)
                    check(f"five_minute_conservation:{method}:{k}:{seed}:{s}",
                          np.array_equal(actual, target), max_absolute_count_error=int(np.max(np.abs(actual-target))))
                    if method == "conditional_60s":
                        check(f"minute_conservation:{k}:{seed}:{s}", np.array_equal(
                            aggregate_counts(generated[s], 60), aggregate_counts(counts[s], 60)*k))
                generated["total"] = generated["code"]+generated["conversation"]
                for s, x in generated.items():
                    base = {"method": method, "multiplier": k, "seed": seed, "service": s}
                    for dt in cfg["analysis_bin_seconds"]:
                        for offset in ([0] if dt == 1 else [0, dt//2]):
                            metrics.append({**base, **arrival_metrics(x, dt, offset)})
                    variation.append({**base, "within_minute_relative_rms_pct": relative_within_block_rms(x)*100,
                                      "within_5min_relative_rms_pct": relative_within_block_rms(x, 300)*100})
                    for day, indices in day_indices.items():
                        y = x[indices]
                        daily_rows.append({**base, "utc_date": day, "seconds": len(y),
                            "request_count": int(y.sum()), "mean_rps": float(y.mean()),
                            "max_1s_rps": int(y.max()), "partial_calendar_day": len(y) != 86400})
                if k == primary and seed == seeds[0]:
                    saved[method] = generated
                    filename = f"facility_arrivals_{method}_K{k}_seed{seed}.npz"
                    np.savez_compressed(output/filename, **{s: generated[s] for s in SERVICES})
                print(f"Compared {method}, K={k}, seed={seed}", flush=True)
    metric_df = pd.DataFrame(metrics)
    var_df = pd.DataFrame(variation)
    diag_df = pd.DataFrame(diagnostics)
    metric_df.to_csv(output/"comparison_metrics.csv", index=False)
    var_df.to_csv(output/"within_block_variability.csv", index=False)
    diag_df.to_csv(output/"source_arrival_diagnostics.csv", index=False)
    pd.DataFrame(daily_rows).to_csv(output/"daily_comparison.csv", index=False)
    pd.concat(demand_frames, ignore_index=True).to_csv(output/"facility_demand_5min.csv", index=False)
    # Source-selected inspection interval, never chosen to favor a synthesis.
    jump = int(np.argmax(np.abs(np.diff(aggregate_counts(counts["code"], 10)/10)))+1)*10
    zoom = min(max(jump-90, 0), n-180)
    ticks = pd.date_range(start, periods=n//300, freq="300s").strftime("%m-%d %H:%M").tolist()
    chart(output/"facility_requests_5min.png", "Facility offered requests: observed five-minute shape",
        f"Illustrative K={primary}; identical five-minute totals for ALL three timing assumptions; not IT power",
        [{"title": "Service-level offered demand", "x": np.arange(n//300), "ticks": ticks,
          "series": {"Code": common_five["code"]*primary/300,
                     "Conversation": common_five["conversation"]*primary/300,
                     "Total": total_five*primary/300}, "unit": "req/s", "xlabel": "Source time (+00:00)", "zero": True}],
        "Source: AzurePublicDataset / DynamoLLM (HPCA 2025). K is not a server count or a 17.39-MW calibration.")
    chart(output/"facility_requests_180s_comparison.png", "Same request totals, different short-time arrival assumptions",
        f"K={primary}; seed={seeds[0]}; 1-s request counts; window starts {(start+pd.Timedelta(seconds=zoom)).isoformat()}",
        [{"title": s.capitalize(), "x": np.arange(180), "series": {m: saved[m][s][zoom:zoom+180] for m in METHODS},
          "unit": "req/s", "xlabel": "Seconds from window start"} for s in SERVICES],
        "Timing assumptions are not validated facility observations. These are arrival-rate changes, not electrical ramps.")
    chart(output/"facility_requests_scale_sensitivity.png", "Short-time variability depends on the scaling assumption",
        "Deviation from each minute's mean / whole-window mean; first seed shown; other seeds saved in CSV",
        [{"title": s.capitalize(), "x": np.asarray(scales), "ticks": [str(k) for k in scales],
          "series": {m: var_df[(var_df.service.eq(s)) & (var_df.method.eq(m)) & (var_df.seed.eq(seeds[0]))]
                              .set_index("multiplier").loc[scales, "within_minute_relative_rms_pct"].to_numpy() for m in METHODS},
          "unit": "%", "xlabel": "Illustrative demand multiplier K", "zero": True} for s in SERVICES],
        "Reduced variability is a consequence of the assumed independent arrivals, NOT evidence of greater realism.")
    for path, digest in before.items():
        check("preserved:"+path, sha256(Path(path)) == digest)
    check("raw_source_five_minute_reconciliation", True)
    check("finite_numeric_metrics", np.isfinite(metric_df[["mean_rps", "cv", "peak_to_mean"]]).all().all())
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(), **provenance,
        "stage": 2, "config": cfg, "source_seconds_per_service": n, "service_mix": mix,
        "primary_multiplier": primary, "seeds": seeds,
        "scale_cases": [{"K (illustrative)": k, "Code mean (req/s)": counts["code"].mean()*k,
                         "Conversation mean (req/s)": counts["conversation"].mean()*k,
                         "Total mean (req/s)": overall/n*k} for k in scales],
        "zoom_start_utc": (start+pd.Timedelta(seconds=zoom)).isoformat(),
        "all_checks_passed": all(c["passed"] for c in checks), "check_count": len(checks),
        "physical_facility_validated": False, "power_profiles_generated": False,
        "instance_allocation_finalized": False, "arrival_model_selected_as_truth": False,
        "source_data_is_forecast_input": False,
        "code_sha256": sha256(Path(__file__)),
        "module_sha256": sha256(ROOT/"src/aidc_facility/facility_arrivals.py"),
        "saved_npz_schema": {"keys": list(SERVICES), "array_type": "int64 offered request counts per left-closed 1-s bin",
             "time_axis": "start_utc + array index seconds; end exclusive; source clock unchanged",
             "not_included": "instance assignments, per-request token marks, completions, queues, power"}}
    (output/"run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output/"verification.json").write_text(json.dumps({"all_passed": summary["all_checks_passed"],
        "scope": "Software/data controls, not physical or statistical validation of the arrival model", "checks": checks},
        ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(checks).to_csv(output/"verification_checks.csv", index=False)
    write_report(output, summary, metric_df, var_df, diag_df, saved)
    print(json.dumps({"output": str(output), "checks": len(checks), "mix": mix}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/facility_arrival_scaling_v1.json")
    parser.add_argument("--raw-dir", type=Path, default=ROOT/"data/raw/azure_inference_2024_full")
    parser.add_argument("--audit-dir", type=Path, default=ROOT/"outputs/azure_long_trace_audit_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/facility_arrival_scaling_v1")
    parser.add_argument("--chunksize", type=int, default=500_000)
    run(parser.parse_args())
