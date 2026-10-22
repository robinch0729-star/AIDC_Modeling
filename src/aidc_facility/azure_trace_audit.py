"""Memory-bounded quality audit of the FULL Azure 2024 invocation traces.

Source: Microsoft AzurePublicDataset, CC BY 4.0; DynamoLLM (HPCA 2025).
This is a descriptive data audit, NOT facility scaling or a power simulation.
No raw record is removed, deduplicated, shifted, interpolated, or synthesized.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]
SOURCE = "https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md"
PAPER = "https://doi.org/10.1109/HPCA61900.2025.00102"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def histogram_summary(hist):
    """Exact empirical nearest-rank quantiles, not a subsample/approximation."""
    if not hist:
        return {"n": 0}
    values = np.array(sorted(hist), dtype=np.int64)
    counts = np.array([hist[int(x)] for x in values], dtype=np.int64)
    n = int(counts.sum())
    cumulative = counts.cumsum()
    result = {"n": n, "min": int(values[0]), "max": int(values[-1]),
              "sum": int(sum(int(v) * int(c) for v, c in zip(values, counts))),
              "zero_count": int(hist.get(0, 0))}
    result["mean"] = result["sum"] / n
    for q in (.5, .9, .95, .99):
        result[f"p{int(q*100)}"] = int(values[np.searchsorted(cumulative, int(np.ceil(q*n)))])
    return result


def audit_file(path, service, output, chunksize=250_000):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if header != SCHEMA:
        raise ValueError(f"Unexpected source columns: {header}")
    counts = Counter()
    token_hist = {name: Counter() for name in SCHEMA[1:]}
    minute_parts = []
    previous_ns = None
    previous_tail = pd.DataFrame(columns=SCHEMA)
    first_ns, last_ns = None, None
    gap_examples, invalid_examples = [], []
    for chunk_no, frame in enumerate(pd.read_csv(path, dtype=str, keep_default_na=False,
                                                skip_blank_lines=False, chunksize=chunksize), 1):
        offset = counts["rows"]
        counts["rows"] += len(frame)
        missing = frame.apply(lambda col: col.str.strip().eq(""))
        for name in SCHEMA:
            counts[f"missing_{name}"] += int(missing[name].sum())
        zone = frame.TIMESTAMP.str.extract(r"(Z|[+-]\d{2}:\d{2})$", expand=False)
        counts["timestamp_timezone_not_explicit"] += int(zone.isna().sum())
        counts["timestamp_nonzero_utc_offset"] += int((zone.notna() & ~zone.isin(["Z", "+00:00", "-00:00"])).sum())
        ts = pd.to_datetime(frame.TIMESTAMP, format="mixed", errors="coerce", utc=True)
        counts["invalid_timestamp"] += int(ts.isna().sum())
        valid_tokens = {}
        for name in SCHEMA[1:]:
            val = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
            finite = np.isfinite(val)
            good = finite & (val >= 0) & (val == np.floor(val)) & (val < 2**53)
            counts[f"invalid_{name}"] += int((~good).sum())
            counts[f"negative_{name}"] += int((finite & (val < 0)).sum())
            counts[f"noninteger_{name}"] += int((finite & (val != np.floor(val))).sum())
            clean = np.where(good, val, 0).astype(np.int64)
            values, frequencies = np.unique(clean[good], return_counts=True)
            token_hist[name].update(dict(zip(values.tolist(), frequencies.tolist())))
            valid_tokens[name] = (clean, good)
        invalid = ts.isna().to_numpy() | ~valid_tokens[SCHEMA[1]][1] | ~valid_tokens[SCHEMA[2]][1]
        counts["rows_with_invalid_timestamp_or_tokens"] += int(invalid.sum())
        if len(invalid_examples) < 20:
            for index in np.flatnonzero(invalid)[:20-len(invalid_examples)]:
                invalid_examples.append({"service": service, "source_row_1based": int(offset+index+2), **frame.iloc[index].to_dict()})

        # Exact within each chunk and its entire previous final timestamp group.
        # Global exactness is certified ONLY if the complete timestamp sequence is sorted.
        joined = pd.concat([previous_tail, frame], ignore_index=True)
        dup = joined.duplicated(SCHEMA).to_numpy()[len(previous_tail):]
        counts["duplicate_rows_detected"] += int(dup.sum())
        previous_tail = joined.loc[joined.TIMESTAMP.eq(joined.TIMESTAMP.iloc[-1])].drop_duplicates(SCHEMA).copy()

        good_ts = ts.notna().to_numpy()
        if good_ts.any():
            ns = ts[good_ts].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
            first_ns = int(ns.min()) if first_ns is None else min(first_ns, int(ns.min()))
            last_ns = int(ns.max()) if last_ns is None else max(last_ns, int(ns.max()))
            extended = np.r_[previous_ns, ns] if previous_ns is not None else ns
            deltas = np.diff(extended)
            counts["timestamp_decreases"] += int((deltas < 0).sum())
            counts["adjacent_equal_timestamps"] += int((deltas == 0).sum())
            for seconds in (1, 10, 60, 300):
                counts[f"interarrival_gaps_gt_{seconds}s"] += int((deltas > seconds*1e9).sum())
            positive = deltas[deltas > 0]
            if len(positive):
                current_min = int(positive.min())
                counts["minimum_positive_interarrival_ns"] = min(counts.get("minimum_positive_interarrival_ns", current_min), current_min)
            for idx in np.argsort(deltas)[-20:]:
                if deltas[idx] > 0:
                    gap_examples.append({"service": service, "start": pd.Timestamp(int(extended[idx]), tz="UTC").isoformat(),
                                         "end": pd.Timestamp(int(extended[idx+1]), tz="UTC").isoformat(), "gap_s": float(deltas[idx]/1e9)})
            gap_examples = sorted(gap_examples, key=lambda x: x["gap_s"], reverse=True)[:20]
            previous_ns = int(ns[-1])
            data = {"minute_start": ts[good_ts].dt.floor("min").to_numpy(), "request_count": np.ones(good_ts.sum(), dtype=np.int64)}
            for name, (values, good) in valid_tokens.items():
                data[name+"_sum_valid"] = values[good_ts]
                data[name+"_invalid_count"] = (~good[good_ts]).astype(np.int64)
            minute_parts.append(pd.DataFrame(data).groupby("minute_start", sort=True).sum())
        if chunk_no % 4 == 0:
            print(f"{service}: audited {counts['rows']:,} rows", flush=True)
    if first_ns is None:
        raise ValueError("No valid timestamps")
    first, last = pd.Timestamp(first_ns, tz="UTC"), pd.Timestamp(last_ns, tz="UTC")
    minutes = pd.concat(minute_parts).groupby(level=0).sum()
    full_index = pd.date_range(first.floor("min"), last.floor("min"), freq="min")
    minutes = minutes.reindex(full_index, fill_value=0).rename_axis("minute_start")
    minutes.insert(0, "service", service)
    # Only brackets supported by observed event endpoints are treated as full bins.
    minutes["full_bin_within_observed_span"] = (minutes.index >= first.ceil("min")) & (minutes.index+pd.Timedelta(minutes=1) <= last)
    minutes["bin_s"] = 60
    minutes["request_rate_rps"] = minutes.request_count / 60
    minutes.to_csv(output / f"{service}_minute_counts.csv")
    summary = {"service": service, **dict(counts), "first_event_utc": first.isoformat(), "last_event_utc": last.isoformat(),
               "observed_span_s": (last-first).total_seconds(), "chronological": counts["timestamp_decreases"] == 0,
               "duplicate_check_global_exact": counts["timestamp_decreases"] == 0 and counts["invalid_timestamp"] == 0,
               "exact_duplicate_definition": "identical raw TIMESTAMP, ContextTokens and GeneratedTokens; no request ID is available",
               "timestamp_duplicate_count": counts["adjacent_equal_timestamps"] if counts["timestamp_decreases"] == 0 else None,
               "token_statistics": {name: histogram_summary(hist) for name, hist in token_hist.items()},
               "valid_timestamp_rows": int(minutes.request_count.sum()),
               "minute_count_reconciles": int(minutes.request_count.sum()) == counts["rows"]-counts["invalid_timestamp"],
               "largest_interarrival_gap_s": gap_examples[0]["gap_s"] if gap_examples else None,
               "full_observed_minutes": int(minutes.full_bin_within_observed_span.sum()),
               "zero_request_full_minutes": int(((minutes.request_count==0)&minutes.full_bin_within_observed_span).sum())}
    hist_rows = [{"service": service, "field": field, "tokens": int(value), "count": int(n)}
                 for field, hist in token_hist.items() for value, n in sorted(hist.items())]
    return summary, minutes, hist_rows, gap_examples, invalid_examples


def common_five_minutes(minutes, summaries):
    start = max(pd.Timestamp(s["first_event_utc"]) for s in summaries).ceil("5min")
    end = min(pd.Timestamp(s["last_event_utc"]) for s in summaries).floor("5min")
    if end <= start:
        raise ValueError("No complete common 5-minute observation window")
    rows = []
    checks = {}
    for service, frame in minutes.items():
        selected = frame[(frame.index >= start) & (frame.index < end)]
        if len(selected) != int((end-start).total_seconds()/60) or not selected.full_bin_within_observed_span.all():
            raise AssertionError("Common range includes an unobserved edge minute")
        cols = ["request_count", "ContextTokens_sum_valid", "GeneratedTokens_sum_valid", "ContextTokens_invalid_count", "GeneratedTokens_invalid_count"]
        five = selected[cols].resample("5min", origin="epoch", closed="left", label="left").sum()
        five.insert(0, "service", service)
        five["bin_s"] = 300
        five["request_rate_rps"] = five.request_count/300
        rows.append(five)
        checks[service] = {"source_minute_requests": int(selected.request_count.sum()),
                           "five_minute_requests": int(five.request_count.sum()),
                           "count_conserved": int(five.request_count.sum()) == int(selected.request_count.sum()),
                           "bins": len(five), "mean_rps": float(five.request_rate_rps.mean()),
                           "min_rps": float(five.request_rate_rps.min()), "max_rps": float(five.request_rate_rps.max())}
    return pd.concat(rows).rename_axis("bin_start_utc"), {"start_inclusive_utc": start.isoformat(),
            "end_exclusive_utc": end.isoformat(), "duration_hours": (end-start).total_seconds()/3600, "services": checks}


def draw_overview(common, output):
    """Scientific line plot using bundled Pillow; no plot styling dependencies."""
    from PIL import Image, ImageDraw, ImageFont
    width, height = 1600, 940
    im = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(im)
    regular = "C:/Windows/Fonts/arial.ttf"
    bold = "C:/Windows/Fonts/arialbd.ttf"
    def font(n, heavy=False):
        return ImageFont.truetype(bold if heavy else regular, n)
    draw.text((100, 25), "Azure 2024: recorded service demand", fill="#142638", font=font(32, True))
    draw.text((100, 70), "Common observation window; 5-minute request counts / 300 s; no facility scaling", fill="#526170", font=font(22))
    colours = {"code": "#2464a5", "conversation": "#bd6516"}
    x0, x1 = 125, 1510
    for index, service in enumerate(("code", "conversation")):
        p = common[common.service.eq(service)]
        top, bottom = 160+index*360, 410+index*360
        y = p.request_rate_rps.to_numpy()
        ymax = float(np.ceil(y.max()/10)*10) or 1
        x = np.linspace(x0, x1, len(y))
        draw.text((x0, top-42), service.capitalize(), fill=colours[service], font=font(25, True))
        for tick in np.linspace(0, ymax, 6):
            yy = bottom-tick/ymax*(bottom-top)
            draw.line((x0, yy, x1, yy), fill="#e1e6ec")
            draw.text((35, yy-12), f"{tick:.0f}", fill="#364554", font=font(19))
        for i in np.linspace(0, len(p)-1, 7).astype(int):
            xx = x[i]
            draw.line((xx, top, xx, bottom), fill="#edf0f3")
            label = p.index[i].strftime("%m-%d %H:%M")
            draw.text((xx-63, bottom+14), label, fill="#364554", font=font(18))
        draw.line(list(zip(x, bottom-y/ymax*(bottom-top))), fill=colours[service], width=2)
        draw.line((x0, top, x0, bottom, x1, bottom), fill="#526170", width=2)
        draw.text((28, top-35), "req/s", fill="#364554", font=font(19))
    draw.text((x0, 855), "Time shown in the source's +00:00 offset. Local service timezone is not established.", fill="#526170", font=font(20))
    draw.text((x0, 890), "Source: Microsoft AzurePublicDataset; DynamoLLM, IEEE HPCA 2025 (CC BY 4.0).", fill="#526170", font=font(20))
    im.save(output / "azure_common_5min_requests.png")


def draw_hourly(hourly, output):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (1550, 890), "white")
    d = ImageDraw.Draw(im)
    def font(n):
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", n)
    d.text((65, 25), "Hourly demand by observed date", fill="#142638", font=font(32))
    d.text((65, 72), "Raw service req/s; each panel uses its own colour scale. Grey = outside full common hours.", fill="#526170", font=font(20))
    for col, service in enumerate(("code", "conversation")):
        p = hourly[hourly.service.eq(service)].copy()
        p["date"] = p.index.strftime("%Y-%m-%d")
        p["hour"] = p.index.hour
        matrix = p.pivot(index="date", columns="hour", values="request_rate_rps").reindex(columns=range(24))
        vmax = float(np.nanmax(matrix.to_numpy())) or 1
        left, top, cellw, cellh = 155+col*740, 170, 23, 58
        d.text((left, 125), f"{service.capitalize()}   (0 to {vmax:.1f} req/s)", fill="#142638", font=font(22))
        for h in range(0, 24, 3):
            d.text((left+h*cellw, top-25), f"{h:02}", fill="#526170", font=font(16))
        for row, (date, values) in enumerate(matrix.iterrows()):
            d.text((left-112, top+row*cellh+17), date[5:], fill="#364554", font=font(20))
            for h, val in enumerate(values):
                if pd.isna(val):
                    colour = (225, 225, 225)
                else:
                    f = float(val/vmax)
                    colour = tuple(int(a+(b-a)*f) for a, b in zip((240,247,253), (31,91,151)))
                xx, yy = left+h*cellw, top+row*cellh
                d.rectangle((xx, yy, xx+cellw-2, yy+cellh-3), fill=colour)
    d.text((65, 785), "Descriptive days only: one short campaign does not establish a repeatable weekday/weekend law.", fill="#526170", font=font(21))
    d.text((65, 830), "Source offset +00:00; Microsoft AzurePublicDataset / DynamoLLM (HPCA 2025).", fill="#526170", font=font(20))
    im.save(output / "azure_hourly_demand_by_date.png")


def write_report(result, daily, token_table, output):
    summaries = result["services"]
    table = pd.DataFrame([{ "서비스": s["service"], "요청 기록 수": f"{s['rows']:,}",
        "최초 요청": s["first_event_utc"], "마지막 요청": s["last_event_utc"],
        "시간 역전": s["timestamp_decreases"], "동일 timestamp 추가 기록": s.get("timestamp_duplicate_count"),
        "완전 동일 행 추가 기록": s["duplicate_rows_detected"],
        "잘못된 시간/token 행": s["rows_with_invalid_timestamp_or_tokens"],
        "최대 요청 간격(s)": round(s["largest_interarrival_gap_s"], 3)} for s in summaries])
    cw = result["common_window"]
    doc = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Azure 장기 요청자료 확보 및 품질검사</title>
<style>body{{font-family:"Malgun Gothic",Arial,sans-serif;max-width:1250px;margin:40px auto;padding:0 24px;line-height:1.75;color:#172b3a}}h1{{font-size:28px}}h2{{margin-top:34px;font-size:22px}}table{{border-collapse:collapse;font-size:13px;width:100%}}th,td{{border-bottom:1px solid #d5dce3;padding:9px;text-align:left}}th{{background:#edf2f6}}img{{width:100%;height:auto}}.note{{background:#f5f7fa;padding:16px}}a{{color:#235f9d}}</style>
<h1>Azure 장기 요청자료 확보 및 품질검사</h1>
<p>단계 ①: 공식 원자료 전체 확보와 관측 품질 확인. 시설 규모 확대, queue 모델, M3·R3 수정은 수행하지 않음.</p>
<h2>1. 자료 확보</h2><ul><li>Code·Conversation 공개 CSV 전체를 다운로드하고 GitHub release의 파일 크기와 SHA-256을 검증.</li>
<li>원자료·기존 180초 표본 보존. 제거·중복 제거·시간 이동·보간·합성·MW 확대 없음.</li>
<li>측정값은 요청 시각과 token 수이며, 실제 전력·완료시간·server 수는 포함하지 않음.*</li></ul>
{table.to_html(index=False,escape=True,border=0)}
<h2>2. 공통 관측기간과 집계</h2><p>{html.escape(cw['start_inclusive_utc'])} ≤ t &lt; {html.escape(cw['end_exclusive_utc'])}: {cw['duration_hours']:.2f}시간.</p>
<p>두 서비스가 함께 관측된 범위에서 완전한 5분 구간만 사용. 요청률 = 구간 요청 수 / 300초.
이 집계는 원자료의 기술통계이며, 전력모델 입력 시간 간격을 확정한 것이 아님.</p>
<img src="azure_common_5min_requests.png" alt="공통 관측기간의 5분 요청률">
<h2>3. 날짜별 요청량</h2>{daily.to_html(index=False,escape=True,border=0,float_format=lambda x:f'{x:,.3f}')}
<img src="azure_hourly_demand_by_date.png" alt="날짜별 시간대 요청률">
<h2>4. Token 통계</h2><p>전체 파일의 유효한 비음수 정수 token 값으로 계산. 분위수는 empirical nearest-rank 방식이며 표본 추정이 아님. 0 token은 자동 오류나 실패로 간주하지 않음.</p>
{token_table.to_html(index=False,escape=True,border=0,float_format=lambda x:f'{x:,.2f}')}
<h2>5. 해석 한계</h2><ul>
<li>요청 간격과 무요청 구간은 관측된 공백이지, 로그 누락이 입증된 구간이 아님. 별도 수집 uptime 정보 없음.</li>
<li>요청 ID가 없으므로 같은 timestamp 또는 세 필드가 같은 행도 중복 수집이라고 확정하지 않으며 삭제하지 않음.</li>
<li>첫·마지막 요청이 실제 관측 시작·종료 시각과 같다는 보장은 없음. 완전 구간은 관측 event 경계 안쪽으로 보수적으로 선택.</li>
<li>+00:00 offset은 보존했으나 실제 서비스 지역·현지 시간대를 추정하지 않음.</li>
<li>한 차례의 여러 날 관측만으로 반복되는 주간 계절성이나 모든 AIDC의 대표성을 확정할 수 없음.</li>
<li>짧은 변동이 Poisson 잡음인지 공통 burst인지 미판정. 초 단위 시설 확대 방식은 다음 단계의 검토 대상.</li>
<li>모든 집계는 원 서비스 규모. 목표 17.39 MW·node 수·NLR 처리용량과 연결하지 않음.</li></ul>
<h2>6. 다음 단계</h2><p>이 원자료로 시설 규모 요청량의 확대·분배 가정을 비교한다. 정상 운전과 혼잡 위험을 판단하기 전에 입력 규모와 변동성의 타당성을 확인한다.</p>
<h2>Reference 및 차용 경계</h2><p>* Microsoft, <a href="{SOURCE}">Azure LLM inference trace 2024</a>, CC BY 4.0.
Jovan Stojkovic et al., <a href="{PAPER}">DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency</a>, IEEE HPCA, 2025.</p>
<p>원자료와 schema는 그대로 사용. 공통기간 선택·1분/5분 집계·품질검사·그림은 본 연구의 전처리이며, 원 논문의 facility 전력모델을 재현했다고 주장하지 않음.</p></html>'''
    (output / "report_ko.html").write_text(doc, encoding="utf-8")


def run_audit(raw_dir, output, chunksize=250_000):
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((raw_dir / "download_manifest.json").read_text(encoding="utf-8"))
    summaries, minutes, all_hist, all_gaps, all_invalid = [], {}, [], [], []
    for source in manifest["sources"]:
        service, path = source["service"], Path(source["path"])
        if path.stat().st_size != source["bytes"] or sha256(path) != source["sha256"]:
            raise ValueError(f"Raw checksum differs from manifest: {service}")
        summary, minute, hist, gaps, invalid = audit_file(path, service, output, chunksize)
        summary["raw_sha256"] = source["sha256"]
        summaries.append(summary)
        minutes[service] = minute
        all_hist.extend(hist)
        all_gaps.extend(gaps)
        all_invalid.extend(invalid)
        print(f"{service}: complete scan, {summary['rows']:,} rows", flush=True)
    common, window = common_five_minutes(minutes, summaries)
    common.to_csv(output / "common_5min_requests.csv")
    hourly_parts, daily_rows, hod_rows = [], [], []
    for service in minutes:
        p = common[common.service.eq(service)]
        hour = p[["request_count"]].resample("h").sum()
        hour["observed_seconds"] = p.bin_s.resample("h").sum()
        hour = hour[hour.observed_seconds.eq(3600)]
        hour["request_rate_rps"] = hour.request_count/3600
        hour["service"] = service
        hourly_parts.append(hour)
        for date, day in p.groupby(p.index.floor("D")):
            duration = int(day.bin_s.sum())
            daily_rows.append({"service": service, "date_utc": str(date.date()), "observed_hours": duration/3600,
                "full_day": duration==86400, "request_count": int(day.request_count.sum()),
                "mean_rps": float(day.request_count.sum()/duration)})
        for h, part in hour.groupby(hour.index.hour):
            hod_rows.append({"service": service, "hour_utc": int(h), "observed_days": len(part),
                "mean_rps": float(part.request_rate_rps.mean()), "min_rps": float(part.request_rate_rps.min()),
                "max_rps": float(part.request_rate_rps.max()), "interpretation": "descriptive observed dates, not validated seasonality"})
    hourly = pd.concat(hourly_parts)
    hourly.to_csv(output / "common_hourly_requests.csv")
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(output / "common_daily_requests.csv", index=False)
    pd.DataFrame(hod_rows).to_csv(output / "hour_of_day_descriptive.csv", index=False)
    tokens = pd.DataFrame([{"service": s["service"], "field": field, **values}
                           for s in summaries for field, values in s["token_statistics"].items()])
    tokens.to_csv(output / "token_statistics.csv", index=False)
    pd.DataFrame(all_hist).to_csv(output / "token_histograms.csv", index=False)
    pd.DataFrame(all_gaps).to_csv(output / "largest_interarrival_gaps.csv", index=False)
    pd.DataFrame(all_invalid, columns=["service", "source_row_1based", *SCHEMA]).to_csv(output / "invalid_row_examples.csv", index=False)
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "stage": "1: full source acquisition and descriptive audit",
        "manifest": str(raw_dir / "download_manifest.json"), "services": summaries, "common_window": window,
        "methods": {"raw_rows_dropped": 0, "deduplication_applied": False, "timestamps_shifted": False,
            "facility_scaling_applied": False, "M3_R3_modified": False, "queue_model_added": False,
            "input_resolution_selected": False, "gap_is_proof_of_missing_data": False,
            "partial_boundary_bins_excluded_from_common_window": True,
            "weekday_pattern_is_validated": False, "token_quantiles": "exact empirical nearest rank"},
        "code_sha256": sha256(Path(__file__)), "source_readme": SOURCE, "reference": PAPER}
    (output / "audit_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_overview(common, output)
    draw_hourly(hourly, output)
    write_report(result, daily, tokens, output)
    print(json.dumps({"services": summaries, "common_window": window}, ensure_ascii=False, indent=2), flush=True)
    return result
