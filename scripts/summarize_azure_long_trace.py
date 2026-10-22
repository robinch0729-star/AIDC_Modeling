"""Add evidence-backed conclusions to the generated stage-1 HTML report."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/azure_long_trace_audit_v1"


def main():
    audit = json.loads((OUT / "audit_summary.json").read_text(encoding="utf-8"))
    checks = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    legacy = json.loads((ROOT / "data/raw/azure_inference_sample/provenance.json").read_text(encoding="utf-8"))
    rows = []
    for s in audit["services"]:
        old = legacy["sources"][s["service"]]
        minute = pd.read_csv(OUT / (s["service"]+"_minute_counts.csv"))
        full = minute[minute.full_bin_within_observed_span]
        rows.append({"service": s["service"], "old_180s_mean_rps": old["window_request_count"]/legacy["duration_s"],
            "whole_file_full_minutes_mean_rps": float(full.request_rate_rps.mean()),
            "common_window_5min_mean_rps": audit["common_window"]["services"][s["service"]]["mean_rps"],
            "context_tokens_old_mean": old["ContextTokens_mean"],
            "context_tokens_full_mean": s["token_statistics"]["ContextTokens"]["mean"],
            "generated_tokens_old_mean": old["GeneratedTokens_mean"],
            "generated_tokens_full_mean": s["token_statistics"]["GeneratedTokens"]["mean"]})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "previous_sample_comparison.csv", index=False)
    count = sum(s["rows"] for s in audit["services"])
    errors = sum(s["rows_with_invalid_timestamp_or_tokens"] for s in audit["services"])
    by_service = {s["service"]: s for s in audit["services"]}
    window = audit["common_window"]
    start = pd.Timestamp(window["start_inclusive_utc"]).strftime("%Y-%m-%d %H:%M")
    end = pd.Timestamp(window["end_exclusive_utc"]).strftime("%Y-%m-%d %H:%M")
    bins = window["services"]["code"]["bins"]
    basic = table[["service", "old_180s_mean_rps", "whole_file_full_minutes_mean_rps", "common_window_5min_mean_rps"]].rename(columns={
        "service": "서비스", "old_180s_mean_rps": "기존 180초 평균(req/s)",
        "whole_file_full_minutes_mean_rps": "전체 파일 완전 1분 구간 평균(req/s)",
        "common_window_5min_mean_rps": "공통 약 5일 평균(req/s)"})
    block = f'''<!-- audit-findings:start --><section class="note"><h2>핵심 결과</h2>
<ul><li>총 {count:,}개 요청 기록 전수 검사. 잘못된 시간/token 행 {errors:,}개.</li>
<li>공통 완전 구간: {start} 이상 {end} 미만(+00:00). 서비스별 5분 bin {bins:,}개.</li>
<li>동일 timestamp 추가 기록은 Code {by_service['code']['timestamp_duplicate_count']:,}개, Conversation {by_service['conversation']['timestamp_duplicate_count']:,}개.
완전히 동일한 행은 각각 {by_service['code']['duplicate_rows_detected']:,}개, {by_service['conversation']['duplicate_rows_detected']:,}개. 기록을 삭제하지 않음.</li>
<li>완전한 1분 구간 중 요청 수 0인 구간은 Code {by_service['code']['zero_request_full_minutes']:,}개, Conversation {by_service['conversation']['zero_request_full_minutes']:,}개.
이 검사가 실제 수집 누락 여부 전체를 확인하는 것은 아님.</li>
<li>독립 행 수·집계·기존 artifact 보존 확인 {len(checks['checks'])}개: {'모두 통과' if checks['all_passed'] else '실패 항목 있음'}.</li>
<li>관측상 Code는 날짜·시간대에 따라 큰 변동을 보이며, Conversation은 다른 형태의 변동을 보임. 원 서비스 요청량이며 node당 요청률이 아님.</li></ul>
<h3>기존 180초 표본과 비교</h3>{basic.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')}
<p>기존 구간은 전체 기간보다 낮은 요청량의 짧은 구간이다. 전체 변동의 대표 표본으로 볼 근거는 부족하지만, 기존 구현 점검용 결과가 무효라는 뜻은 아니다.</p>
<p>자료 확보·관측값 검사는 완료. 시설 규모 확대 방식, NLR 처리상태 대응, 전력 모델 신뢰도는 다음 단계에서 별도로 검증해야 한다.</p></section><!-- audit-findings:end -->'''
    report = OUT / "report_ko.html"
    content = report.read_text(encoding="utf-8")
    if "<!-- audit-findings:start -->" in content:
        start = content.index("<!-- audit-findings:start -->")
        end = content.index("<!-- audit-findings:end -->", start)+len("<!-- audit-findings:end -->")
        content = content[:start]+content[end:]
    content = content.replace("<h2>1. 자료 확보</h2>", block+"<h2>1. 자료 확보</h2>")
    report.write_text(content, encoding="utf-8")
    print(table.to_string(index=False))
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
