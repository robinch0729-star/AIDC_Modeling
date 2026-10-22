"""Run phase 3 diagnostics, preserving all load generators and previous results."""
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
from aidc_facility.nlr_processing_diagnostic import (SERVICES, WINDOWS, sha256, scan_server,
    diagnose_runs, align_logged_intervals, finite_diagnostics, residence_screen)

RATE = "inference_online_rate_llama3_70b"
FINITE = "inference_online_finite_llama3_70b"
NAMES = {SERVICES[0]: "InstructCoder", SERVICES[1]: "Llama2-data"}
COLORS = ["#2465a5", "#c65c28", "#299078", "#9c4ba4"]


def plot(path, title, subtitle, panels, footer):
    from PIL import Image, ImageDraw, ImageFont
    width, height = 1650, 175 + 375*len(panels)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    def font(size, bold=False):
        return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", size)
    draw.text((120, 23), title, fill="#173348", font=font(30, True))
    draw.text((120, 69), subtitle, fill="#4c606d", font=font(20))
    for index, p in enumerate(panels):
        left, right = 130, 1550
        top, bottom = 183+index*375, 420+index*375
        draw.text((left, top-70), p["title"], fill="#18374e", font=font(23, True))
        all_y = np.concatenate([np.asarray(s[2], float) for s in p["series"]])
        all_y = all_y[np.isfinite(all_y)]
        lo, hi = float(all_y.min()), float(all_y.max())
        pad = max((hi-lo)*.08, .01)
        ymin, ymax = (0 if p.get("zero") else lo-pad), hi+pad
        xmin = min(np.min(s[1]) for s in p["series"])
        xmax = max(np.max(s[1]) for s in p["series"])
        if "xlim" in p:
            xmin, xmax = p["xlim"]
        for y in np.linspace(ymin, ymax, 6):
            py = bottom-(y-ymin)/(ymax-ymin)*(bottom-top)
            draw.line((left, py, right, py), fill="#dfe6eb")
            label = f"{y:,.0f}" if ymax>200 else f"{y:.1f}"
            draw.text((left-15, py), label, anchor="rm", fill="#415562", font=font(18))
        for x in np.linspace(xmin, xmax, 7):
            px = left+(x-xmin)/(xmax-xmin or 1)*(right-left)
            draw.line((px, top, px, bottom), fill="#ecf0f3")
            draw.text((px, bottom+12), f"{x:.0f}", anchor="mt", fill="#415562", font=font(18))
        for j, (label, x, y) in enumerate(p["series"]):
            color = COLORS[j % len(COLORS)]
            lx = left+j*(1420/max(3, len(p["series"])))
            draw.line((lx, top-29, lx+25, top-29), fill=color, width=3)
            draw.text((lx+34, top-40), label, fill="#304653", font=font(18))
            x, y = np.asarray(x, float), np.asarray(y, float)
            points = []
            for xv, yv in zip(x, y):
                if not np.isfinite(yv):
                    if len(points)>1:
                        draw.line(points, fill=color, width=2)
                    points = []
                    continue
                point = (left+(xv-xmin)/(xmax-xmin or 1)*(right-left), bottom-(yv-ymin)/(ymax-ymin)*(bottom-top))
                points.append(point)
                if p.get("dots"):
                    draw.ellipse((point[0]-3, point[1]-3, point[0]+3, point[1]+3), fill=color)
            if len(points)>1:
                draw.line(points, fill=color, width=2)
        draw.line((left, top, left, bottom, right, bottom), fill="#5b6d79", width=2)
        draw.text((left, top-2), p["unit"], anchor="lb", fill="#415562", font=font(17))
        draw.text(((left+right)/2, bottom+42), p["xlabel"], anchor="mt", fill="#415562", font=font(18))
    draw.text((120, height-36), footer, fill="#4c606d", font=font(17))
    image.save(path)


def tables_and_summary(windows, boundaries):
    primary = windows[windows.window.eq("primary_60_240")].copy()
    onset, estimates, sensitivity = [], [], []
    for service in SERVICES:
        p = primary[primary.service.eq(service)].sort_values("configured_rps")
        b = boundaries[boundaries.service.eq(service)]
        def first(frame, col):
            found = frame[frame[col].gt(0)]
            return float(found.configured_rps.min()) if len(found) else None
        onset.append(dict(service=service, first_sampled_waiting_full_run_rps=first(b, "full_run_waiting_max"),
                          first_sampled_waiting_primary_rps=first(p, "waiting_max"),
                          high_rate_81_100_runs=int(p.configured_rps.ge(81).sum()),
                          high_rate_81_100_waiting_runs=int((p.configured_rps.ge(81)&p.waiting_max.gt(0)).sum())))
        for name in WINDOWS:
            # Predeclared high-rate band; excludes demand-limited and startup snapshots.
            q = windows[(windows.service.eq(service)) & (windows.window.eq(name)) &
                windows.configured_rps.ge(60) & windows.waiting_positive_all_snapshots &
                windows.metric_count.ge(6) & windows.max_metric_gap_s.le(20)]
            estimates.append(dict(service=service, window=name, eligible_runs=len(q),
                conditional_departure_median_rps=float(q.conditional_departure_rps.median()),
                conditional_departure_min_rps=float(q.conditional_departure_rps.min()),
                conditional_departure_max_rps=float(q.conditional_departure_rps.max()),
                conditional_endpoint_median_rps=float(q.conditional_endpoint_departure_rps.median()),
                configured_rate_min=float(q.configured_rps.min()), configured_rate_max=float(q.configured_rps.max())))
        for tolerance in [.01, .02, .05]:
            records = []
            for rate in p.configured_rps:
                q = windows[(windows.service.eq(service)) & windows.configured_rps.eq(rate)]
                passed = all(residence_screen(row, tolerance) for row in q.to_dict("records"))
                if passed:
                    records.append(int(rate))
            prefix = 0
            while prefix+1 in records:
                prefix += 1
            sensitivity.append(dict(service=service, relative_drift_tolerance=tolerance,
                all_four_windows_pass_count=len(records), contiguous_pass_prefix_max_rps=prefix,
                passing_configured_rates=",".join(map(str, records))))
    return primary, pd.DataFrame(onset), pd.DataFrame(estimates), pd.DataFrame(sensitivity)


def make_figures(output, primary, windows, intervals):
    panels = []
    for col, title, unit in [("gpu_mean_kw", "Measured GPU power only (4 GPU channels)", "kW"),
            ("running_mean", "Running: time-averaged sampled scheduler stock", "requests"),
            ("waiting_max", "Waiting: maximum sampled stock in primary window", "requests"),
            ("generation_tps_snapshot_mean", "Logged generation throughput (not request throughput)", "tokens/s")]:
        series = [(NAMES[s], primary[primary.service.eq(s)].configured_rps,
                   primary[primary.service.eq(s)][col]) for s in SERVICES]
        panels.append(dict(title=title, unit=unit, series=series, zero=True, xlabel="Configured offered rate (request/s per benchmark instance)"))
    plot(output/"processing_state_by_rate.png", "NLR online-rate: power, scheduler stocks and token throughput",
         "200 runs | primary window: run start +60 to +240 s | no serving-capacity assumption", panels,
         "GPU power is not full IT/PCC power. Waiting onset is a sampled observation, not an SLO or sustainable-capacity limit.")
    panels = []
    for service in SERVICES:
        for field, label in [("running_at_end", "Running"), ("waiting_at_end", "Waiting")]:
            series = []
            for rate in [30, 40, 50, 90]:
                p = intervals[(intervals.service.eq(service)) & intervals.configured_rps.eq(rate) &
                              intervals.relative_end_s.ge(60) & intervals.relative_end_s.lt(240)]
                series.append((f"{rate} request/s", p.relative_end_s-60, p[field]))
            panels.append(dict(title=f"{NAMES[service]} | {label}", unit="requests", zero=True, dots=True,
                series=series, xlim=(0,180), xlabel="Time since start of 180-s diagnostic window (s)"))
    plot(output/"scheduler_state_examples_180s.png", "Why zero Waiting does not prove steady operation",
         "Raw scheduler snapshots, approximately every 10 s; lines connect observations, not a reconstructed fine-time queue", panels,
         "A growing Running stock is also unfinished work. No processing response time is fitted from these curves.")
    panels = []
    for service in SERVICES:
        series = []
        for window, label in [("primary_60_240", "60-240 s"), ("early_60_150", "60-150 s"), ("late_150_240", "150-240 s")]:
            p = windows[(windows.service.eq(service)) & windows.window.eq(window) & windows.configured_rps.ge(40)].copy()
            p.loc[~p.waiting_positive_all_snapshots, "conditional_departure_rps"] = np.nan
            series.append((label, p.configured_rps, p.conditional_departure_rps))
        panels.append(dict(title=NAMES[service], unit="conditional departures (request/s)", series=series, dots=True,
                          xlabel="Configured offered rate (request/s per benchmark instance)"))
    plot(output/"conditional_departure_sensitivity.png", "Conditional request-departure proxy, NOT validated serving capacity",
         "configured rate - slope(Running + Waiting) | only windows with Waiting > 0 at every snapshot", panels,
         "Requires actual admitted rate = configured rate, matched scheduler boundary, and no cancellation/other departures.")


def report(output, summary, primary, onset, estimates, sensitivity, finite_grouped):
    def table(frame):
        frame = frame.copy()
        for col in frame.columns:
            if pd.api.types.is_string_dtype(frame[col].dtype):
                frame[col] = frame[col].str.replace(",", ", ", regex=False)
        return frame.round(3).to_html(index=False, border=0, na_rep="—")
    selection = primary[primary.configured_rps.isin([10, 20, 30, 40, 42, 44, 50, 70, 90, 100])]
    selection = selection[["service", "configured_rps", "running_mean", "waiting_max", "unfinished_slope_rps", "conditional_departure_rps", "conditional_residence_proxy_s"]]
    selection.columns = ["Dataset", "설정 요청률(req/s)", "평균 Running", "최대 Waiting", "미완료 증가율(req/s)", "조건부 이탈률(req/s)", "조건부 체류시간(s)"]
    cap = estimates[estimates.window.eq("primary_60_240")].copy()
    cap = cap[["service", "eligible_runs", "conditional_departure_median_rps", "conditional_departure_min_rps", "conditional_departure_max_rps"]]
    fin = summary["finite_benchmark"]
    references = [
      ("NLR [3] — Vercellino et al., Measurement of Generative AI Workload Power Profiles for Whole-Facility Data Center Infrastructure Planning (2026), §2.3.2 / §4.2.", "https://arxiv.org/html/2604.07345v1#S2.SS3.SSS2"),
      ("vLLM v0.11.0 — LoggingStatLogger: token rate intervals and scheduler-state logging.", "https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/metrics/loggers.py"),
      ("vLLM v0.11.0 — benchmark serve: configured arrivals, burstiness, completed/duration and E2EL.", "https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/benchmarks/serve.py"),
      ("Little, J. D. C. (1961), A Proof for the Queuing Formula: L = λW, Operations Research 9(3), 383–387.", "https://doi.org/10.1287/opre.9.3.383")]
    ref_html = "".join(f'<li id="ref{i}">*{i}. <a href="{url}">{html.escape(label)}</a></li>' for i, (label,url) in enumerate(references,1))
    document = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>③ NLR 처리 상태 진단</title>
<style>body{{font-family:"Malgun Gothic",Arial,sans-serif;max-width:1250px;margin:36px auto;padding:0 25px;color:#203747;line-height:1.8}}h1{{font-size:29px}}h2{{font-size:23px;margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #d9e2e8;text-align:left}}th{{background:#eef3f7}}img{{max-width:100%;height:auto}}.note{{padding:16px 22px;background:#f0f5f8;border-left:4px solid #317498}}.formula{{padding:13px 18px;font-family:Cambria,serif;font-size:22px;background:#f5f6f8}}a{{color:#216699}}li{{margin:6px 0}}code{{overflow-wrap:anywhere}}</style>
<h1>③ NLR 처리 상태 진단</h1>
<p class="note"><strong>완료 범위:</strong> 원시 server 로그 재파싱, 200개 online-rate 실행의 처리 상태·전력 시간 정렬, 처리율의 조건부 추정, finite 결과의 지연 지표 확인.<br>
<strong>미수행:</strong> queue 모델 구축, M3/R3 수정, instance 수·배정 비율 결정, facility 부하 재생성. 이 단계의 추정값을 자동으로 생성기에 입력하지 않았다.</p>
<h2>1. 핵심 결론</h2><ul>
<li><strong>Waiting=0만으로 정상상태를 판정하면 안 된다.</strong> 예를 들어 40 req/s에서 두 dataset 모두 Waiting이 없지만 Running은 증가한다. 미완료 작업은 Running+Waiting으로 진단해야 한다.</li>
<li>전체 run에서 처음 관측된 Waiting은 InstructCoder 42, Llama2-data 44 req/s이다. <strong>이 값은 보장된 처리용량이 아니라 해당 길이의 실험·10초 로그에서 관측된 시작점</strong>이다. 분석 창을 바꾸면 시작점도 달라진다.</li>
<li>60–100 req/s 중 지속 Waiting 조건의 미완료 증가율로 조건부 이탈률을 계산했다. 아래 수치는 직접 측정한 완료율도, SLO를 만족하는 확정 처리용량도 아니다.</li>
<li>전력 포화와 처리용량을 분리해야 한다. GPU 전력은 먼저 완만해지는 반면 Running·Waiting은 계속 증가할 수 있다.</li>
<li>R3 고율 benchmark 81–100 req/s의 40개 조건 모두 180초 분석 창에서도 Waiting이 관측된다. 고율 검증 결과를 정상 서비스 품질 검증으로 해석하지 않는다.</li></ul>
{table(cap)}<p>범위는 설정 요청률별 관측 차이이며 통계적 신뢰구간이 아니다. early·late·alternative 창의 별도 결과는 아래에 제공한다.</p>
<h2>2. 원자료와 식별 범위</h2><ul>
<li>online-rate: dataset 2종 × 요청률 1–100 = 200개 run, 원시 server metric {summary['raw_scan']['parsed_metrics']:,}행. 로그의 vLLM 버전은 0.11.0, model은 Llama-3.1-70B-Instruct, tensor parallel 4이다.*1,*2</li>
<li>수치는 GPU 한 개가 아니라 해당 <strong>4-GPU serving benchmark instance 전체</strong>에 대한 요청률이다. 조건마다 1회이고 요청률을 순차 증가시킨 실행이므로 독립 반복실험의 신뢰구간이나 모든 hardware에 통용되는 성능으로 해석하지 않는다.</li>
<li>입력은 configured request rate. num_prompts=rate×300, burstiness=100000. 이는 평균 간격이 거의 일정한 생성 설정이며 Poisson burstiness=1과 다르다.*3 실제 scheduler admission timestamps를 기록한 것은 아니다.</li>
<li>run lifecycle 약 320초. 기존 M3와 같은 run start+60초부터 +240초 미만을 주요 진단 창으로 두고, 60–150·150–240·90–270초를 비교한다. 실제 요청 시작·종료와 lifecycle 경계는 동일하지 않을 수 있다.</li>
<li>Running·Waiting은 logger가 가진 최신 scheduler 상태이며, token/s는 직전 로깅 구간의 처리 token 수/경과시간이다.*2 이 둘을 모두 ‘10초 최대값’이라고 부르는 것은 잘못이다. 최대값은 이번 분석에서 snapshot들을 모아 별도로 계산했다.</li>
<li>online-rate는 save_result=False, save_detailed=False이며 per-request 완료·E2EL 자료가 없다. 날짜 timezone과 장치 간 시계 동기화도 검증되지 않았으므로 wall-clock 정렬은 인과적 전이시간 식별이 아니다.</li>
<li>GPU 4개 채널의 합과 CPU reported-domain 합을 분리 저장. CPU package/core 중복 경계가 미해결이므로 그래프는 GPU만 표시하며 전체 IT 또는 PCC 실측으로 부르지 않는다.</li></ul>
{table(onset)}
<img src="processing_state_by_rate.png" alt="요청률별 GPU 전력 Running Waiting 및 token 처리량">
<h2>3. 미완료 작업 증가율과 조건부 이탈률</h2>
<div class="formula">B(t) = R(t) + Q(t)<br>ΔB = A<sub>scheduler</sub> − C<sub>completed</sub> − X<sub>other exits</sub><br>β̂ = slope(B(t)); &nbsp; μ̃ = r<sub>configured</sub> − β̂</div>
<p>R=Running, Q=Waiting. 첫 두 식은 동일한 scheduler 경계에 대한 상태 정의·수지식이지 새 부하 모델이 아니다. 마지막 식은 <strong>실제 admitted rate≈설정 rate, 취소·실패·기타 이탈≈0</strong>을 가정하여 수지식에 OLS 추세를 적용한 우리 진단 방식이다. 기존 문헌의 확정 capacity 식을 그대로 차용한 것이 아니다. 실제 scheduler 밖에서 대기하는 요청은 B에 포함되지 않는다.</p>
<p>모든 관측 창에 μ̃를 계산하지만, 비혼잡 조건에서는 공급 가능한 최대량이 아니라 현재 수요에 가까운 값이 된다. 처리능력 후보 요약에는 사전에 정한 60–100 req/s 중 모든 snapshot에서 Waiting&gt;0인 창만 사용했다. 로그 gap≤20초, snapshot≥6 조건도 적용했다. 임의로 μ를 한 값에 고정하지 않았다.</p>
{table(selection)}
<p>예: 40 req/s에서 Running이 쌓이면 μ̃는 40보다 작다. Waiting이 없어도 이미 작업 체류가 늘어나는 상태일 수 있다. 반대로 90 req/s에서는 들어온 요청을 모두 처리했다는 뜻이 아니며, 많은 양이 미완료 작업으로 누적된다.</p>
<img src="scheduler_state_examples_180s.png" alt="180초 창 Running Waiting 원시 관측 비교">
<img src="conditional_departure_sensitivity.png" alt="조건부 처리율 추정의 창 선택 민감도">
{table(estimates)}
<p class="note">일부 run 종료 근처에서 큰 미완료 stock이 갑자기 0으로 떨어진다. 이 현상은 완료·취소·종료 정리 중 무엇인지 INFO 로그만으로 구분할 수 없다. <strong>이를 자연적인 backlog 소진 응답으로 간주하거나 전력 시정수를 추정하지 않았다.</strong> 종료 경계는 주요 60–240초 추정 창 밖에 두고 별도 CSV로 보관했다. 에러/abort 문자열 부재도 취소가 없었다는 증거는 아니다.</p>
<h2>4. 체류시간: 추정할 수 있는 것과 없는 것</h2>
<div class="formula">L = λW &nbsp; (*4) &nbsp; → &nbsp; W̃ = mean(Running) / r<sub>configured</sub></div>
<p>Little의 법칙은 같은 시스템 경계의 장기 평균 재고·유입률·체류시간 관계다.*4 이번 W̃는 실제 admission rate를 몰라 <strong>조건부 참고값</strong>이다. Waiting=0, 미완료 증가율과 시작–끝 증가율의 절댓값이 모두 설정 rate의 2% 이하인 조건을 <strong>네 개 창에서 모두 만족한 run</strong>에만 표시했다. 2%는 우리 진단 편의 기준이며 안정성 증명이나 문헌의 SLO 기준이 아니다. 1%·5% 민감도도 제공한다. 극저율에서는 작은 stock 변화도 비율상 커져 screen에서 제외될 수 있으므로 탈락을 불안정 운전 판정으로 해석하지 않는다.</p>
<p>평균 stock은 첫~마지막 snapshot 사이 trapezoid 평균이며 전체 180초의 빈 구간을 외삽하지 않았다. 미측정 10초 내부 상태를 실제 복원한 것은 아니다. 10·20·30 req/s에서 수 초 규모의 참고값이 나와도 <strong>5분 전력 에너지 오차가 몇 %라는 뜻은 아니며</strong>, 이를 전력 전이 시정수로 쓰지 않는다. 2% screen을 통과한 단일 창도 진정한 정상상태라고 보장하지 않는다.</p>
{table(sensitivity)}
<p>연속 통과 상한은 ‘보수적 안정용량’이 아니다. 짧은 창의 통계 잡음과 시작 상태에 민감하므로 전체 통과 요청률 목록을 함께 기록했다.</p>
<h2>5. finite 자료에서 직접 확인할 수 있는 지연 정보</h2>
<p>metadata {fin['metadata_runs']}개 중 결과 {fin['result_runs']}개. 누락 ID: {', '.join(fin['missing_result_ids'])}. 기록된 completed는 모두 1000이며, mean E2EL은 run별 {fin['mean_e2el_s_min']:.2f}–{fin['mean_e2el_s_max']:.2f}초이다. 이 지표는 요청 송신부터 응답 완료까지 client 측 지연이고, online-rate의 scheduler 내부 W̃와 다르다.*3</p>
<p>finite는 hf-output-len=256·512·1024 설정과 1000개 요청 실험이다. online-rate의 hf_output_len=None·장시간 주입과 동일 운전조건이 아니다. 따라서 finite E2EL·완료율을 online-rate의 처리용량으로 복사하지 않았다. [3]의 예시 instance capacity 역시 latency 조건을 사용한 finite 기반 산정과 현재 지속 주입의 안정성 문제를 구분해야 한다.*1</p>
<p class="formula">finite request throughput = completed / benchmark duration &nbsp; (*3)</p>
<p>이 항등식을 1024개 기록에서 재계산하여 검증했다. 처리 token/s를 Azure 평균 token 수로 나누어 request/s로 바꾸는 추정은 하지 않았다. 각 finite 조건별 latency·실측 평균 출력 token 수는 <a href="finite_grouped.csv">별도 CSV</a>에 저장했다.</p>
<h2>6. 현재 연구에 적용할 결론과 다음 순서</h2><ol>
<li><strong>기존 모델 해석 수정:</strong> M3는 offered rate별 평균 전력 관계다. 과부하 구간도 포함하므로 해당 전력이 서비스 품질을 만족한다는 뜻이 아니다. R3 고율 benchmark에는 미완료·대기 누적이 있었다는 운전 맥락을 붙인다.</li>
<li><strong>시설 규모와 instance 배정은 아직 확정하지 않는다.</strong> Azure 39:61은 관측 표본의 요청 구성비이고 node 수 비율이 아니다. 이번 NLR 수치를 Azure token 분포가 다른 요청에 그대로 적용하는 데에도 workload 대응 가정이 남는다.</li>
<li><strong>④ 최소 미완료 작업 이월 모델은 민감도 비교부터:</strong> 이번 조건부 처리율과 창별 차이를 가정 범위로 사용하되 실제 capacity라고 고정하지 않는다. 내부 짧은 시간축에서 인과적으로 요청을 처리하고 출력만 5분으로 평균한다. 구체적인 내부 간격은 수렴성 비교로 정한다.</li>
<li>대기뿐 아니라 처리 중 작업도 미완료 상태에 포함하고, 실측 조건과 동떨어진 M3 입력 재정의를 피한다. 이월 상태→전력의 연결은 별도 검증 항목으로 남긴다.</li>
<li>확정 capacity·SLO가 필요하면 같은 serving 설정에서 admitted/completed/cancelled timestamp, token 수, TTFT/E2EL을 전력과 동기 기록한 부하상승·하강 실험이 추가로 필요하다. 현재 로그로는 임의의 τ나 확정 μ를 식별했다고 주장하지 않는다.</li></ol>
<h2>7. 재현성과 파일</h2><ul>
<li><a href="window_diagnostics.csv">800개 창 진단</a> · <a href="run_boundary_diagnostics.csv">200개 run 경계</a> · <a href="aligned_logger_power.csv">동일 logger 구간의 token·전력 정렬</a></li>
<li><a href="raw_server_metrics.csv">원시 로그 재파싱 결과</a> · <a href="source_manifest.json">원본/기존 결과 hash</a> · <a href="verification.json">자동 검증</a> · <a href="independent_verification.json">독립 재계산 검증</a></li>
<li>실행: <code>python scripts/run_nlr_processing_diagnostic.py</code>. 기존 raw·M3·R3·② 산출물은 읽기만 하고 별도 output 폴더에 저장했다.</li></ul>
<h2>References / 방법론 차용 구분</h2><ol>{ref_html}</ol>
<p>직접 차용: NLR 로그·실험조건, vLLM 지표 정의, Little 관계식. 연구 맞춤 적용: 분석 창, stock 추세 회귀, configured rate 대입, 저변동 screen, 창별 민감도. 이들은 물리적 처리용량이나 전이모델이 검증됐다는 뜻이 아니다.</p></html>'''
    (output/"report_ko.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/nlr_processing_diagnostic_v1")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    raw = ROOT.parent/"NLR_dataset/00_raw_datasets"
    old = ROOT/"outputs/nlr_transition_audit_v1"
    checks, manifest = [], []
    def check(name, passed, detail=None):
        checks.append(dict(check=name, passed=bool(passed), detail=detail))
        if not passed:
            raise AssertionError(name + ": " + str(detail))
    coverage = {r["relative_path"]: r for r in json.loads((old/"coverage.json").read_text())}
    protected = [ROOT/"outputs/r3_high_rate_benchmark"/f for f in
        ("train_validation_m3_parameters.csv", "train_validation_residual_parameters.csv")]
    protected += sorted((ROOT/"outputs/facility_arrival_scaling_v1").glob("*"))
    protected = [p for p in protected if p.is_file()]
    before = {str(p): sha256(p) for p in protected}
    print("Rescanning original vLLM server log...", flush=True)
    metrics, scan, evidence = scan_server(raw/RATE/"vllm_server.log", 2026)
    check("server_metric_parser_complete", scan["unparsed_metric_markers"] == 0)
    check("raw_metric_time_monotonic", metrics.timestamp.is_monotonic_increasing and not metrics.timestamp.duplicated().any())
    for rel, cov in coverage.items():
        if rel.startswith(RATE+"/") or rel in [FINITE+"/log.json", FINITE+"/log_vllm.json"]:
            actual = scan["sha256"] if rel == RATE+"/vllm_server.log" else sha256(raw/rel)
            check("source_hash:"+rel, actual == cov["sha256"])
            manifest.append(dict(path=str(raw/rel), sha256=actual, matches_prior_audit=True))
    previous = pd.read_csv(old/f"{RATE}_server_metrics.csv")
    previous["timestamp"] = pd.to_datetime(previous.timestamp)
    check("prior_metric_count", len(previous) == len(metrics))
    for col in previous.columns:
        check("reparsed_equals_prior:"+col, np.array_equal(previous[col].to_numpy(), metrics[col].to_numpy()))
    runs = pd.read_csv(old/f"{RATE}_runs.csv", dtype={"id": str})
    raw_runs = pd.DataFrame(json.loads((raw/RATE/"log.json").read_text()))
    check("200_unique_runs", len(runs)==200 and runs.id.nunique()==200)
    check("raw_run_ids_equal", runs.id.tolist()==raw_runs.id.tolist())
    check("raw_run_start_equal", pd.to_datetime(runs.run_start).equals(pd.to_datetime(raw_runs.start_time.str.replace("_", "T"))))
    check("raw_run_end_equal", pd.to_datetime(runs.run_end).equals(pd.to_datetime(raw_runs.end_time.str.replace("_", "T"))))
    check("raw_run_rates_equal", runs.request_rate.equals(raw_runs.request_rate))
    check("raw_run_datasets_equal", runs["dataset-path"].equals(raw_runs["dataset-path"]))
    run_starts = pd.to_datetime(runs.run_start)
    run_ends = pd.to_datetime(runs.run_end)
    check("runs_do_not_overlap", bool((run_starts.iloc[1:].to_numpy() >= run_ends.iloc[:-1].to_numpy()).all()))
    check("configured_prompt_count", bool((raw_runs["num-prompts"] == raw_runs.request_rate*300).all()))
    check("configured_burstiness", bool(raw_runs.burstiness.eq(100000).all()))
    for s in SERVICES:
        check("rate_coverage:"+s, sorted(runs[runs["dataset-path"].str.endswith(s)].request_rate.tolist())==list(range(1,101)))
    power = pd.read_csv(old/f"{RATE}_power_1s_diagnostic.csv")
    power["timestamp"] = pd.to_datetime(power.timestamp)
    for p in (old/f"{RATE}_runs.csv", old/f"{RATE}_power_1s_diagnostic.csv", old/f"{RATE}_server_metrics.csv"):
        manifest.append(dict(path=str(p), sha256=sha256(p), role="previous derived audit input"))
    print("Diagnosing 200 runs x 4 windows; aligning token logger intervals with power...", flush=True)
    windows, boundaries = diagnose_runs(runs, metrics, power)
    aligned = align_logged_intervals(runs, metrics, power)
    check("800_window_rows", len(windows)==800)
    check("all_windows_enough_snapshots", windows.metric_count.ge(6).all())
    check("positive_time_coverage", windows.observed_span_s.gt(0).all())
    check("power_window_count", windows.power_seconds.eq(windows.requested_window_s).all())
    check("slope_additivity", np.allclose(windows.unfinished_slope_rps, windows.running_slope_rps+windows.waiting_slope_rps))
    check("conditional_flow_identity", np.allclose(windows.conditional_departure_rps+windows.unfinished_slope_rps, windows.configured_rps))
    check("no_cross_run_alignment", aligned.relative_end_s.gt(aligned.interval_s).all())
    primary, onset, estimates, sensitivity = tables_and_summary(windows, boundaries)
    check("report_full_run_waiting_onset", onset.first_sampled_waiting_full_run_rps.tolist()==[42.,44.])
    check("report_high_rate_waiting_context", onset.high_rate_81_100_waiting_runs.tolist()==[20,20])
    meta = pd.DataFrame(json.loads((raw/FINITE/"log.json").read_text()))
    results = pd.DataFrame([json.loads(line) for line in (raw/FINITE/"log_vllm.json").read_text().splitlines() if line.strip()])
    finite, finite_grouped, finite_summary = finite_diagnostics(meta, results)
    check("finite_throughput_identity", finite_summary["max_throughput_identity_error"] < 1e-10)
    check("finite_missing_ids_expected", finite_summary["missing_result_ids"]==["000687", "000709"])
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), stage="3: NLR processing-state diagnostic",
        raw_scan=scan, primary_window_s=[60,240], window_count=len(windows), run_count=len(runs),
        finite_benchmark=finite_summary, waiting_onset=onset.to_dict("records"),
        conditional_capacity_sensitivity=estimates.to_dict("records"), low_drift_screen=sensitivity.to_dict("records"),
        no_generator_changes=True, no_mu_or_transition_model_locked=True,
        assumptions=["Configured request rate substitutes for unobserved scheduler admissions only in conditional proxies",
                     "No independent online-rate per-request completion/latency measurements",
                     "No cancellation/other departures assumed for conditional completion interpretation; not verified",
                     "Naive wall-clock alignment, not verified device clock synchronization",
                     "GPU-only figure; CPU domain overlap unresolved"])
    for name, frame in [("raw_server_metrics",metrics), ("window_diagnostics",windows), ("run_boundary_diagnostics",boundaries),
            ("aligned_logger_power",aligned), ("waiting_onset",onset), ("conditional_capacity_sensitivity",estimates),
            ("residence_screen_sensitivity",sensitivity), ("finite_results",finite), ("finite_grouped",finite_grouped),
            ("raw_log_evidence",pd.DataFrame(evidence))]:
        frame.to_csv(output/f"{name}.csv", index=False, encoding="utf-8-sig")
    print("Writing figures and Korean report...", flush=True)
    make_figures(output, primary, windows, aligned)
    report(output, summary, primary, onset, estimates, sensitivity, finite_grouped)
    for p, digest in before.items():
        check("preserved:"+str(Path(p).relative_to(ROOT)), sha256(Path(p)) == digest)
    for name, obj in [("run_summary",summary), ("source_manifest",manifest),
                      ("verification",dict(passed=all(c["passed"] for c in checks), count=len(checks), checks=checks))]:
        (output/f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(output=str(output), checks=len(checks), waiting_onset=summary["waiting_onset"],
        primary_conditional_estimates=estimates[estimates.window.eq("primary_60_240")].to_dict("records")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
