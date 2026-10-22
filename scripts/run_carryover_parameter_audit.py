"""Reproducible stage 4-1 audit. Writes only a new output directory."""
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
from aidc_facility.carryover_parameter_audit import build_audit, PRIMARY
from aidc_facility.nlr_processing_diagnostic import sha256, SERVICES


def verify_raw_windows(windows, raw):
    """Independent recalculation from raw-derived snapshots, no diagnosis helper."""
    raw = raw.copy()
    raw["timestamp"] = pd.to_datetime(raw.timestamp)
    max_mean_error = max_slope_error = 0.
    for row in windows.itertuples():
        g = raw[raw.timestamp.between(pd.Timestamp(row.first_snapshot), pd.Timestamp(row.last_snapshot))]
        if len(g) != row.metric_count:
            raise AssertionError(f"Snapshot count mismatch: {row.run_id}/{row.window}")
        t = (g.timestamp - g.timestamp.iloc[0]).dt.total_seconds().to_numpy(float)
        r, q = g.running_requests.to_numpy(float), g.waiting_requests.to_numpy(float)
        mean = np.sum((r[:-1]+r[1:])*np.diff(t)/2)/(t[-1]-t[0])
        slope = np.polyfit(t, r+q, 1)[0]
        max_mean_error = max(max_mean_error, abs(mean-row.running_mean))
        max_slope_error = max(max_slope_error, abs(slope-row.unfinished_slope_rps))
        if int(r.max()) != row.running_max or int(q.max()) != row.waiting_max:
            raise AssertionError("Stock maximum mismatch")
        if abs(row.configured_rps-slope-row.conditional_departure_rps) > 1e-8:
            raise AssertionError("Conditional departure arithmetic mismatch")
    if max_mean_error > 1e-8 or max_slope_error > 1e-8:
        raise AssertionError("Raw-snapshot recalculation failed")
    return {"independently_recomputed_windows": len(windows),
        "max_running_mean_error": max_mean_error, "max_stock_slope_error": max_slope_error,
        "note": "Rechecks derived raw snapshots, not a new scan of the 290 MB server log"}


def report(a):
    def table(rows):
        return pd.DataFrame(rows).to_html(index=False, border=0, float_format=lambda v: f"{v:.3f}")
    names = {SERVICES[0]: "InstructCoder", SERVICES[1]: "Llama2-data"}
    token_evidence = a["token_transfer_evidence"]
    az_output = {r["service"]: r["mean"] for r in token_evidence["azure_statistics"]
                 if r["field"] == "GeneratedTokens"}
    finite_output = {r["service"]: r["mean_output_tokens"]
                     for r in token_evidence["nlr_finite_summary"] if r["hf-output-len"] == 256}
    mix = a["arrival_contract"]["source_mix"]
    summary, anchor_rows, conflicts, sensitivity = [], [], [], []
    for name, s in a["services"].items():
        m, l, c = s["conditional_departure"], s["low_load_residence"], s["concurrency"]
        summary.append({"서비스": names[name], "조건부 이탈률 중심값(req/s)": m["central_rps"],
            "창별 최솟값": m["window_envelope_min_rps"], "창별 최댓값": m["window_envelope_max_rps"],
            "관측 Running 최대": int(c["observed_peak_running"]),
            "10·20·30 rps 체류 proxy 중앙값(s)": l["anchor_median_s"]})
        for r in l["curve"]:
            if r["configured_rps"] in [10,20,30,35]:
                anchor_rows.append({"서비스": names[name], "설정 요청률(req/s)": r["configured_rps"],
                    "평균 Running": r["running_mean"], "조건부 체류시간(s)": r["conditional_residence_proxy_s"],
                    "창별 최소(s)": r["window_min_s"], "창별 최대(s)": r["window_max_s"]})
        for r in s["slot_consistency"]:
            conflicts.append({"서비스": names[name], "가정 slot 수": r["hypothetical_slots"],
                "저부하 체류 proxy(s)": r["low_load_lifetime_proxy_s"],
                "단순 slot 모델의 포화 처리율(req/s)": r["implied_full_pool_rps"],
                "NLR 조건부 이탈률(req/s)": r["conditional_high_load_departure_rps"],
                "해당 slot에서 필요한 체류시간(s)": r["lifetime_needed_at_full_pool_s"]})
        for r in m["windows"]:
            sensitivity.append({"서비스": names[name], "분석 창": r["window"],
                "기존 유효 run 수": r["eligible_runs"], "기존 중앙값(req/s)": r["median_conditional_departure_rps"],
                "공통 run 수": r["common_cohort_runs"], "공통 run 중앙값(req/s)": r["common_cohort_median_rps"]})
    return f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>④-1 최소 이월 모델 파라미터 검토</title><style>
body{{font-family:'Malgun Gothic',sans-serif;max-width:1180px;margin:40px auto;padding:0 24px;color:#213143;line-height:1.75}}
h1{{font-size:28px}}h2{{margin-top:36px;font-size:22px}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}}
th,td{{padding:10px;border-bottom:1px solid #dce3ea;text-align:right}}th:first-child,td:first-child{{text-align:left}}
th{{background:#edf2f6}}.note{{background:#fff5df;padding:14px 18px}}.eq{{font-family:Cambria,serif;font-size:20px;padding:12px;background:#f4f6f8}}
a{{color:#225d92}}img{{width:100%;height:auto}}.small{{font-size:13px;color:#586777}}</style>
<h1>④-1 최소 미완료 작업 이월 모델: 파라미터 근거·정합성 검토</h1>
<p>작성: {html.escape(a['created_utc'])} · 범위: NLR online-rate, Llama-3.1-70B, TP=4 serving 단위. 개별 GPU의 처리용량이 아님.</p>
<p class="note"><strong>결론:</strong> 파라미터 근거표와 일관성 검사를 완료했다. 다만 <strong>하나의 고정 처리시간과 고정 slot 수를 갖는 모델을 NLR 재현 모델로 확정할 수 없다.</strong>
부하에 따라 처리 중 체류가 달라지는 구조를 다음 단계에서 검토해야 한다. 기존 M3·R3, 요청자료, 시설 IT 생성기는 수정하지 않았다.</p>
<h2>1. 최소 모델에서 구분할 값</h2>
<p class="eq">B = Q + R<br>Q<sub>k+1</sub> = Q<sub>k</sub> + A<sub>k</sub> − J<sub>k</sub><br>
R<sub>k+1</sub> = R<sub>k</sub> + J<sub>k</sub> − C<sub>k</sub></p>
<p>Q: 대기 요청, R: 처리 중 요청, A: 해당 구간의 scheduler 유입, J: 처리 시작, C: 완료 요청 수.
위 식은 다른 이탈·재진입이 없을 때의 일반적인 재고 보존식이다. [43]의 원문 식으로 인용하지 않는다.
도착·시작·완료 시점이 없으면 이 식만으로 J와 C를 결정할 수 없다. Running은 매 순간 모든 요청이 동시에 GPU 연산 중이라는 뜻도 아니다.*3</p>
<p>[43] §3.3 Eq. (6)의 active-request 수를 R로 표기하면 다음과 같다.*1</p>
<p class="eq">R(t) = Σ<sub>i</sub> 1{{t<sub>i</sub><sup>start</sup> ≤ t &lt; t<sub>i</sub><sup>end</sup>}}</p>
<p>원형의 활성 구간 집계 정의를 참고한다. 원문의 batch size 64, TTFT/TBT 수치, GMM·BiGRU 전력 생성기는 이 단계에서 가져오지 않았다.
NLR에서는 request별 시작·완료 및 token–latency 쌍이 없어 원문 수명분포 파라미터를 그대로 추정할 수 없다.</p>
<h2>2. 데이터에서 얻을 수 있는 값과 한계</h2>{table(summary)}
<p>중심값은 60–240초 창에서 설정 요청률 ≥60 req/s, 모든 snapshot에 Waiting&gt;0, 유효 snapshot ≥6개, 최대 간격 ≤20초인 run의 중앙값이다.
범위는 네 분석 창의 조건부 추정값 전체 min–max이며 <strong>신뢰구간이 아니다.</strong> 같은 run의 겹치는 창을 독립 표본처럼 취급하지 않는다.</p>
<p class="eq">μ̃ = r<sub>configured</sub> − slope(R + Q)</p>
<p>이는 재고 보존식에 실제 유입률 대신 설정 요청률을 넣은 <strong>우리의 조건부 추정</strong>이다.
유입률=설정률, 취소 등 다른 이탈 없음이라는 가정이 필요하다. 실측 완료율, 확정 처리용량, SLO 충족 한계가 아니다.</p>
<p>Running 최대 1,024는 관측값이다. 원시 로그의 초기 설정에서 max_num_seqs 명시값을 확보하지 못했으므로 scheduler 설정 K=1,024로 확정하지 않았다.
로그의 max_num_batched_tokens=8,192는 token 예산이며 request slot 수와 다르다.</p>
<h2>3. 처리 중 체류시간은 부하에 따라 달라진다</h2>
<p class="eq">W̃<sub>R</sub>(r) = R̄(r) / r<sub>configured</sub></p>
<p>Little의 평균 재고–유량 관계 L=λW를 조건부로 적용한 proxy다.*2
Waiting=0이고 미완료 증가율이 작은 조건을 네 개 창 모두 통과한 경우만 사용했다. 2% screen은 연구자 진단 기준이지 문헌의 안정성/SLO 기준이 아니다.
10·20·30 req/s는 서비스 간 비교용 anchor이며, 다른 통과 조건도 JSON에 모두 보존했다.</p>
{table(anchor_rows)}<img src="parameter_consistency.png" alt="조건부 체류시간과 고정 slot 처리율의 정합성 비교">
<p>40 req/s는 두 서비스 모두 이 screen을 통과하지 못한다. 따라서 그 조건의 Running/r을 정상상태 서비스시간이라고 사용하지 않았다.
이 proxy만으로 분포 형태, 잔여 처리시간 또는 prefill/decode 단계를 식별할 수 없다. 평균만 맞는 deterministic·exponential·lognormal 모델도 이월 결과가 다를 수 있다.</p>
<h2>4. 고정 처리시간·slot 수를 독립적으로 지정하면 생기는 모순</h2>
<p class="eq">μ<sub>slot</sub> = K / E[S] &nbsp; (고정된 독립 처리 slot이 모두 차 있는 단순 모형)*2</p>
{table(conflicts)}
<p>1,024 slot과 저부하 약 3초의 체류시간을 그대로 결합하면 수백 req/s를 처리하는 모델이 되어 NLR의 조건부 이탈률과 큰 차이가 난다.
반대로 K/μ̃의 수십 초를 모든 요청에 적용하면 저부하의 약 3초와 맞지 않는다.
<strong>이는 constant-lifetime slot 모형의 이 조합이 맞지 않는다는 진단이지, 실제 NLR 서비스시간을 수십 초로 측정했다는 뜻이 아니다.</strong>
[43]의 64 slot을 복사하는 것도 NLR의 관측 Running 약 1,000개를 재현하지 못한다. 별도 매핑 없는 64 slot NLR 대체안은 채택하지 않는다.</p>
<h2>5. 시간 창에 따른 처리율 차이</h2>{table(sensitivity)}
<p>기존 창별 통계는 유효 run 집합도 바뀔 수 있다. 이번에는 네 창에 공통으로 포함된 run만으로도 중앙값을 다시 계산해 시간 창 효과와 표본 구성 효과를 구분했다.
중앙값 하나로 μ를 확정하지 않고 다음 단계의 민감도 범위로 남긴다.</p>
<h2>6. Azure 요청자료와 NLR 파라미터를 연결하는 경계</h2>
<ul><li>Azure code/conversation을 NLR InstructCoder/Llama2-data에 연결하는 것은 서비스 유형 대응이다. token 길이·처리시간이 같다는 가정이 아니다.</li>
<li>공통 약 120시간 자료의 요청 비율 {mix['code']['fraction']*100:.3f}%/{mix['conversation']['fraction']*100:.3f}%를 유지한다. 구간별 비율을 강제로 39:61로 고정하지 않는다.</li>
<li>전체 Azure 원자료의 평균 생성 token은 code {az_output['code']:.2f}, conversation {az_output['conversation']:.2f}이다.
NLR finite에서 출력 상한 256인 경우 평균 생성 token은 각각 약 {finite_output[SERVICES[0]]:.2f}, {finite_output[SERVICES[1]]:.2f}이다.
서로 다른 모집단이며 finite를 online-rate로 대체할 수는 없지만, 서비스 이름만으로 처리비용을 동일시할 근거가 없음을 보여준다.</li>
<li>④-2 첫 비교는 Azure의 <strong>도착 시각 패턴 + NLR workload 구성</strong>이라는 합성 시나리오로 한정하는 것이 권장된다. Azure token 정보까지 실제로 재현했다고 표시하지 않는다.</li>
<li>시설 확대계수·instance 수·분배 정책은 아직 선택하지 않았다. ①·②의 100/1,000/3,000은 요청 규모 민감도 후보이지 서버 수가 아니다.</li></ul>
<h2>7. 다음 ④-2로 넘길 작업</h2>
<ol><li>한 NLR serving 단위의 상태부터 비교: queue-only 기준안과 부하 의존 처리 체류를 허용한 상태 모델 후보. 저부하 Running, 고부하 Running·Waiting 증가율을 동시에 검사.</li>
<li>Q·R 상태는 구간 사이에 유지. replay 시작값은 관측값, 합성 입력은 warm-up과 초기 상태 민감도. 종료 시 누적 작업을 임의로 완료 처리하지 않음.</li>
<li>상태 비교에서는 R3를 제외해 효과를 분리. 요청 지연을 ESS–TES와 같은 제어 유연성으로 취급하지 않음.</li>
<li>내부 계산 간격 0.1/0.5/1초는 수렴성 후보이며 NLR 상태의 실측 해상도가 아님. 최종 출력은 300초. 5분에 한 번만 큐를 갱신하지 않음.</li>
<li>M3 입력은 기존의 설정/offered request rate 의미를 유지. 완료율·Running·backlog를 바로 대입하지 않음. 상태→전력 연결은 후속 별도 검증.</li></ol>
<p class="note">이번 단계의 완료 의미: <strong>사용 가능한 조건부 값, 상충하는 조합, 미식별 파라미터를 확정</strong>했다.
완성된 이월 모델이나 전력 전이모델을 확보했다는 뜻은 아니다. 외부 benchmark 자료나 동일 설정의 추가 측정 없이 token별 수명분포를 유일하게 결정할 수 없다.</p>
<h2>Reference와 차용 범위</h2>
<p>*1 [43] G. Wilkins, F. Kazhamiaka, R. Rajagopal, <a href="https://arxiv.org/html/2603.18383v1#S3.SS3">From Servers to Sites: Compositional Power Trace Generation of LLM Inference for Infrastructure Planning</a>, arXiv:2603.18383v1, 2026, §3.3, Eqs. (4)–(6).
활성 요청의 구간 집계와 수명 기반 surrogate 구조 참고. NLR 수치 보정 및 부하 의존성 검토는 본 연구의 작업.</p>
<p>*2 J. D. C. Little, <a href="https://doi.org/10.1287/opre.9.3.383">A Proof for the Queuing Formula: L = λW</a>, Operations Research, 9(3), 383–387, 1961.
평균 재고–유량–체류시간 관계 차용. 설정 요청률 대입, 저변동 screen, slot 일관성 검사는 본 연구의 조건부 적용이며 원문의 추정 절차가 아님.</p>
<p>*3 <a href="https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/metrics/loggers.py">vLLM v0.11.0 metrics/loggers.py</a>.
Running·Waiting은 최근 scheduler 상태이고 token throughput은 logging 구간 평균. 약 10초 snapshot으로 sub-second 상태 이력을 복원하지 않음.</p>
<p>[3] NLR 원자료의 online-rate 및 finite 로그. 기존 ③ 분석의 source_manifest와 raw snapshot을 재사용했고 800개 창을 독립 재계산했다.
검증 수치·입력 해시는 <a href="verification.json">verification.json</a>과 <a href="source_manifest.json">source_manifest.json</a>, 전체 근거는 <a href="parameter_contract.json">parameter_contract.json</a>에 보존했다.</p>
</html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/carryover_parameter_audit_v1")
    parser.add_argument("--refresh", action="store_true", help="Refresh only an existing stage-4-1 output from this builder")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        marker = out/"parameter_contract.json"
        if not args.refresh or not marker.is_file():
            raise FileExistsError(f"Refusing to overwrite an existing audit: {out}")
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("stage") != "4-1: carryover parameter evidence and consistency audit":
            raise ValueError("Refresh requires this builder's previous stage-4-1 output")
    d = ROOT/"outputs/nlr_processing_diagnostic_v1"
    paths = {"windows": d/"window_diagnostics.csv", "raw": d/"raw_server_metrics.csv",
        "finite": d/"finite_results.csv", "phase3_manifest": d/"source_manifest.json",
        "azure_tokens": ROOT/"outputs/azure_long_trace_audit_v1/token_statistics.csv",
        "arrival_summary": ROOT/"outputs/facility_arrival_scaling_v1/run_summary.json"}
    # Preserve previous model source and result artifacts; hash before and after.
    protected = list((ROOT/"src/nlr_data").glob("*.py"))
    for folder in ("facility_arrival_scaling_v1", "nlr_processing_diagnostic_v1"):
        protected.extend((ROOT/"outputs"/folder).glob("*"))
    protected = sorted(set(p for p in protected if p.is_file()))
    before = {str(p): sha256(p) for p in protected}
    windows = pd.read_csv(paths["windows"], dtype={"run_id": str})
    raw = pd.read_csv(paths["raw"])
    finite = pd.read_csv(paths["finite"], dtype={"id": str})
    tokens = pd.read_csv(paths["azure_tokens"])
    arrivals = json.loads(paths["arrival_summary"].read_text(encoding="utf-8"))
    audit = build_audit(windows, finite, tokens, arrivals)
    audit["created_utc"] = datetime.now(timezone.utc).isoformat()
    checks = verify_raw_windows(windows, raw)
    if len(windows) != 800 or windows.run_id.nunique() != 200:
        raise AssertionError("Unexpected phase-3 scope")
    out.mkdir(parents=True, exist_ok=args.refresh)
    def dump(name, value):
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    dump("parameter_contract.json", audit)
    dump("source_manifest.json", [{"role": role, "path": str(p), "sha256": sha256(p)} for role,p in paths.items()]
        + [{"role": "builder", "path": str(Path(__file__)), "sha256": sha256(Path(__file__))},
           {"role": "analysis", "path": str(ROOT/"src/aidc_facility/carryover_parameter_audit.py"),
            "sha256": sha256(ROOT/"src/aidc_facility/carryover_parameter_audit.py")}])
    from run_nlr_processing_diagnostic import plot
    series = []
    for service in SERVICES:
        curve = audit["services"][service]["low_load_residence"]["curve"]
        series.append(("InstructCoder" if service == SERVICES[0] else "Llama2-data",
            [r["configured_rps"] for r in curve], [r["conditional_residence_proxy_s"] for r in curve]))
    plot(out/"parameter_consistency.png", "Phase 4-1: a single service lifetime is not established",
        "Conditional Running / configured-rate proxy; only runs passing all four low-drift screens",
        [{"title": "Observed conditional residence increases with offered load", "series": series,
          "xlabel": "Configured request rate (req/s per serving unit)", "unit": "Seconds", "dots": True}],
        "Not direct per-request latency, a power time constant, or an identified service-time distribution.")
    (out/"report_ko.html").write_text(report(audit), encoding="utf-8")
    unchanged = all(sha256(Path(p)) == digest for p,digest in before.items())
    if not unchanged:
        raise AssertionError("Pre-existing artifacts changed")
    checks.update({"all_checks_passed": True, "protected_file_count": len(before),
        "protected_files_unchanged": unchanged, "run_count": 200,
        "strict_json_export_passed": True, "scope_is_parameter_audit_only": True})
    dump("verification.json", checks)
    print(json.dumps({"output": str(out), "verification": checks,
        "summary": {name: {"mu_proxy": s["conditional_departure"]["central_rps"],
            "low_load_lifetime_proxy": s["low_load_residence"]["anchor_median_s"],
            "slot_consistency": s["slot_consistency"]} for name,s in audit["services"].items()}},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
