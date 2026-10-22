# 시간가변 IT 부하 실험 실행 안내

## 구현 범위

- 기존 M3/R3 parameter와 기존 출력은 수정하지 않음.
- Azure 공개 요청 도착 자료의 첫 공통 180초 구간을 사용함. 시설 규모에 맞춘 요청 **크기**는 연구 가정이고, 시간축은 그대로 유지함.
- 10초 요청 집계 + 0.1초 R3 생성. 10초 경계의 M3 즉시 이동은 실제 고속 전이응답이 아님.
- 가동 노드 수 고정, 균등 분배, 독립 residual, state 연속. queue·autoscaling·cooling·PCC·DFL은 아직 미포함.
- NLR-equivalent 6,487개 단위는 가상의 기준 시설이며 실측 서버 대수가 아님.
- 전력 채널 경계와 CPU package/core 합산의 미확인 사항은 [감사 문서](time_varying_it_model_audit.md) 참조.

## 파일 역할

| 파일 | 역할 |
|---|---|
| `src/aidc_facility/arrival_trace.py` | 제한된 HTTP byte range로 같은 날짜의 요청 표본 취득, 출처/hash 기록, 재집계 검증 |
| `src/aidc_facility/time_varying_it.py` | 요청 분배·지원범위 검사·연속 R3·노드 합산·경계 ramp 진단 |
| `scripts/fetch_inference_traffic_sample.py` | 최초 공개자료 취득 CLI; 인터넷 접근 필요 |
| `scripts/run_time_varying_it.py` | 저장된 원자료 검증 후 고정수요 B와 시간가변수요 C 실행 |
| `tests/test_arrival_trace.py`, `tests/test_time_varying_it.py` | 자료 집계·요청 보존·상태 지속·경계 탐지 등 단위 테스트 |

## 실행

PowerShell에서 저장소 루트를 작업 디렉터리로 사용:

```powershell
Set-Location D:\DC_modeling\aidc-dfl-doe-research
# 최초 자료 취득에만 필요. 이미 저장된 sample은 재사용 가능.
& D:\DC_modeling\NLR_dataset\.venv\Scripts\python.exe scripts/fetch_inference_traffic_sample.py

# cached sample의 SHA256 및 원자료 재집계를 검사하고 실행
& D:\DC_modeling\NLR_dataset\.venv\Scripts\python.exe scripts/run_time_varying_it.py

# 전체 테스트
& D:\DC_modeling\NLR_dataset\.venv\Scripts\python.exe -m pytest tests -q
```

- 새 실험은 `--output-dir outputs/새_실험명`으로 분리하여 기존 결과 보존.
- `--scenarios`는 **같은 요청 자료에 대한 residual 실현 개수**. 관측 traffic 표본 개수가 아님.
- 생성기 입력 조건을 바꾸려면 서비스 설정·지원범위·목표 기준점의 의미를 먼저 확인. 모델 보정범위 밖 rate와 무요청 구간은 자동 extrapolation하지 않음.
- 기존 A는 저장된 결과를 읽기만 함. B/C는 같은 노드 수·평균 요청량·분배 방식으로 비교.
- 17.391304 MW는 B의 기준 M3 입력 운전점을 설계하는 수치. C와 개별 실현값을 사후 정규화하지 않음.

## 출력

기본 출력: `outputs/time_varying_it_v1/`

- `report_ko.md`: 결과·그림·수식과 reference 각주·남은 검증
- `traffic_to_power.png`: 요청량 → M3 운전점 → 합산 residual
- `boundary_ramp_diagnostic.png`: 집계 경계 포함·제외 ramp 구분
- `comparison_summary.csv`, `scenario_metrics.csv`: A/B/C 비교
- `facility_profiles.csv`, `example_node_profiles.csv`: 시설·예시 단위 시계열
- `request_dispatch.csv`, `request_accounting.csv`: 전체/배분/초과 요청량
- `it_profiles.mat`: MATLAB에서 읽을 수 있는 MW 전력과 초 시간축
- `run_config.json`: source와 parameter hash·가정·검증 범위

출력과 raw 자료는 기존 `.gitignore`에서 제외되며, 이 작업은 commit/push를 수행하지 않음.

## 핵심 해석 제한

1. Azure 데이터는 전력 자료가 아니라 요청 시각·token 수 자료. 현재 token 수는 진단용으로만 기록됨.
2. 서비스별 시간 모양을 NLR prompt 조건에 연결한 시나리오이며 동일 모델·hardware·token 분포라는 주장이 아님.
3. 표본은 처음 공통 180초로 미리 정했으며, 높은 변동 구간을 검색해 선택하지 않았음. 대표성·일반성 검증은 아직 없음.
4. 0.1초 출력은 residual 보정 시간간격. 요청에 따른 전력 전이응답이 0.1초 정확도로 검증됐다는 뜻이 아님.
5. Node power는 비유계 Gaussian-mixture tail을 포함. 명판 참고값 초과를 별도 기록하며 물리 hard cap으로 임의 clipping하지 않음.

## Reference

- *NLR [3], §2.3.2/§4.2: 요청 분배·측정 node power 합산 구조. https://arxiv.org/html/2604.07345v1
- *[43], §3: 요청 기반 전력 생성·시설 합산 관점. M3+R3 자체의 식을 제공하지 않음. https://arxiv.org/html/2603.18383v1
- *Azure LLM Inference Dataset 2024 (CC BY 4.0), DynamoLLM, IEEE HPCA 2025: https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md
