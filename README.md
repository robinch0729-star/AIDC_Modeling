# AIDC DFL–DOE Research

배전계통에 연계된 AI 데이터센터(AIDC)의 부하·불확실성 정보를 추정하고, 이를 DSO의 Dynamic Operating Envelope(DOE) 산정과 ESS–TES 기반 local dispatch에 활용하기 위한 연구 코드 저장소입니다.

## 배전계통 기준 및 시간가변 배경부하

- 주 테스트베드 계열은 기존 계획의 **Modified IEEE 33-bus**입니다. SimBench **계통** 검토는 대안 이력으로 보존하며, 이번 단계에서는 SimBench의 **부하 시간 형태만** 별도로 차용합니다.
- 사용자 선택 전압은 **12.66 kV**입니다. 원형 공급선로와 전원 1.0 p.u.에서는 현재 AIDC 규모를 연결할 때 전압·전류 제약을 만족하지 못했습니다. 이전 33 kV 후보는 비교 이력으로만 보존합니다.
- 사용자가 **여유 보강안을 기준 계통, 최소 보강안을 비교 시나리오**로 확정했습니다. 현재 정의는 `configs/ieee33_operating_reference_v1.json`, 상태 구분은 `configs/grid_testbed_registry.json`이 우선합니다. 이전 `selected_candidates.json`의 확정 전 상태는 당시 기록으로 보존합니다. 연구 기준 확정이며 실제 접속설계 승인은 아닙니다.
- Bus 1–2만 변경하고 전원 전압을 1.04 p.u.로 고정합니다. 기준 계통의 등가 R/X는 **0.021333/0.0396 Ω**, C는 **1170 nF**, 허용전류는 **1.785 kA**입니다. 비교 계통은 **0.0215/0.0558 Ω**, **980 nF**, **1.524 kA**입니다. 제조사 자료·1 km 가정·병렬회선의 원래 선정 근거는 이전 보고서에 보존합니다. 기준 계통에 −10% 정격 저감을 몰래 적용하지 않고 별도 민감도로 구분합니다.
- 공칭전압, 나머지 선로 R/X, 기존 시설 PCC 프로파일, ESS–TES 사양·스케줄을 유지합니다. 나머지 선로 600 A 및 상위 공급한계 40 MVA는 연구 가정이며, 40 MVA는 모델링된 변압기 정격이 아닙니다.
- 결과는 `outputs/ieee33_reinforcement_v1/report_ko.html`에 있습니다. AIDC 연결 상태로 후보를 선정하고, 1,438개 5분 운전점·충전 전력 포락선·AIDC 정지 및 경부하를 검사했습니다. 포락선은 지속 가능한 SOC 스케줄이 아닙니다.
- ‘최소’는 변경 구간 수 → 회선 수 → 단면적 → 전원 전압의 사전 선정 규칙에 따른 제한된 후보군 최소입니다. 실제 시공·보호·N−1 설계, 전역 최소비용, DOE 및 새로운 계통 연계 저장장치 최적화 결과를 의미하지 않습니다.
- 시간가변 배경부하는 `configs/ieee33_time_varying_background_v1.json`으로 별도 정의합니다. SimBench **1-MV-urban--0-sw의 수요 형태만** 차용하며 계통은 IEEE33을 유지합니다. 2016년 5월 둘째 일요일부터 5일의 15분 자료를 5분 유지값으로 변환하여 기존 AIDC 1,438구간과 연결합니다. 실제 동시·동일지역 계측이 아닌 합성 결합 시나리오입니다.
- 기존 배경 P/Q의 **기간 평균 유지**, **기간 최대 유지**, **평균 유지 + 6시간 상대 이동**을 사전 정의하여 비교합니다. 모든 기존 모선에 동일 계수를 적용하고 AIDC는 제외합니다. 기존 모선별 비중·역률은 유지하지만 실제 공간적 다양성이나 독립적인 5분 변동은 재현하지 않습니다.
- 결과 및 자료 출처는 `outputs/ieee33_time_varying_background_v1/report_ko.html`에 기록합니다. 시간가변 조건에서의 실패는 그대로 보고하며 전압·정격·저장장치를 자동 재조정하지 않습니다. 기존 25.5 MW 스케줄 재생은 새로운 DOE·운전 최적화가 아닙니다.

재현 명령(기존 NLR 환경과 분리된 `.venv-grid` 사용):

```powershell
& .venv-grid/Scripts/python.exe scripts/assess_ieee33_time_varying_background_v1.py
& .venv-grid/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_ieee33_time_varying_background_v1.py --junitxml=outputs/ieee33_time_varying_background_v1/test_results.xml
& .venv-grid/Scripts/python.exe scripts/report_ieee33_time_varying_background_v1.py
```

이전 정적 보강 후보 검토의 재현 명령(확정 전 이력만 재생):

```powershell
& .venv-grid/Scripts/python.exe scripts/screen_ieee33_reinforcement_v1.py
& .venv-grid/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_ieee33_reinforcement_v1.py --junitxml=outputs/ieee33_reinforcement_v1/test_results.xml
& .venv-grid/Scripts/python.exe scripts/report_ieee33_reinforcement_v1.py
```

## 연구 프레임워크

```text
AIDC gray-box 모델 및 NLR workload
→ DFL 기반 부하·불확실성 정보추정
→ AIDC가 추정정보를 DSO에 전달
→ DSO OPF 기반 PCC import DOE 할당
→ Energy Router의 ESS–TES local dispatch
→ OPAL-RT 기반 DOE 지령추종 검증
```

현재까지 **NLR 데이터셋 탐색·특성추출, online-inference 정상상태 수식 비교,
M3 residual·OU innovation 진단, MAD-free R1/R2/R3 validation 비교 및 고율
holdout benchmark**를 완료했습니다. 빠른 ramp와 비정규 변동 재현을 목적으로
R3 Jump-OU를 고정했으며, Train+Validation 재보정 파라미터를 사용하는 최종
M3+R3 IT 부하 프로파일 생성기를 구현했습니다.

## 디렉터리 구조

```text
aidc-dfl-doe-research/
├─ configs/       # 데이터 경로와 실험 설정
├─ notebooks/     # 탐색적 데이터 분석 및 결과 시각화
├─ src/           # 재사용 가능한 전처리·특성추출·모델 코드
├─ tests/         # 단위 테스트
├─ outputs/       # 생성 결과(기본적으로 Git에서 제외)
├─ .gitignore
├─ README.md
└─ requirements.txt
```

## 데이터셋

NLR 데이터셋은 용량과 원본 이력 보존 문제로 이 저장소에 포함하지 않습니다. 기본 로컬 위치는 다음과 같습니다.

```text
D:\DC_modeling\NLR_dataset
```

코드는 원본 데이터를 수정하지 않고 읽기 전용으로 사용하며, 전처리 데이터와 분석 결과는 별도 출력 폴더에 저장하는 것을 원칙으로 합니다.

## Python 환경

Python 3.12 사용을 권장합니다.

```powershell
cd D:\DC_modeling\aidc-dfl-doe-research

py -V:3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

설치 확인:

```powershell
python -c "import numpy, pandas, pyarrow, scipy; print('Environment ready')"
```

## 개발 진행 현황

1. NLR 폴더·파일 메타데이터 인덱싱 — 완료
2. 스키마, 시간축, 단위 및 결측치 점검 — 완료
3. workload 유형별 시계열 정규화와 특성 추출 — 완료
4. condition-aware split과 정상상태 M1–M3 비교 — 완료
5. M3 residual·OU innovation 구조 진단 — 완료
6. MAD-free R1 white noise / R2 OU / R3 Jump-OU validation 비교 — 완료
7. 목적 기반 R3 고정 및 81–100 request/s 고율 benchmark — 완료
8. 재보정된 M3와 고정 R3 기반 node-level IT profile generator — 완료
9. NLR [3] DIPLOEE 흐름 기반 bottom-up facility IT 합산 — 완료
10. 냉각·UPS·보조부하 결합 및 AIDC PCC 모델 구성 — 예정
11. DFL 기반 정보추정과 DSO OPF–DOE 연계 — 예정

신규 residual 진단·선정 절차는
`notebooks/08a_nlr_residual_diagnostics.ipynb`와
`notebooks/08b_nlr_residual_model_selection.ipynb`에 있습니다. 기존
`08_nlr_residual_model_comparison.ipynb`와 `09_nlr_workload_aggregation.ipynb`는
MAD 기반 R3를 사용한 구버전 산출물이므로 최종 결과로 사용하지 않습니다.
현재 집합화 절차는 `notebooks/09_nlr_workload_aggregation_v2.ipynb`에 있으며,
출처가 불명확한 `sqrt(rho)` 식 대신 equivalent workload-group profile의 직접
가중합과 명시적인 독립·부분동기·완전동기 시나리오를 사용합니다.

## 기존 equivalent M3+R3 IT profile generator

다음 예시는 `InstructCoder`, 90 request/s 조건에서 평균 20 MW의 equivalent IT
부하 시나리오 100개를 생성합니다.

```powershell
python scripts\generate_final_it_profiles.py `
  --prompt-dataset InstructCoder `
  --request-rate 90 `
  --scenario-count 100 `
  --duration-s 180 `
  --interval-s 0.1 `
  --seed 42 `
  --target-it-mean-mw 20 `
  --output-dir outputs\final_it_profile_generator\InstructCoder_90rps_20MW
```

출력에는 long/wide CSV, MATLAB/Simulink용 `it_profiles.mat`, 생성 설정,
파라미터 snapshot, 그림 및 방법론 보고서가 포함됩니다. R3는 0.1초 간격에서
보정되었으므로 현재 생성기도 `--interval-s 0.1`만 허용합니다. 20 MW scaling은
냉각·UPS·보조부하가 제외된 equivalent IT 부하이며 PCC 총부하가 아닙니다.
현재 facility 결과에는 이 단순 MW 배율 확대를 사용하지 않습니다.

## NLR [3] 기반 bottom-up facility IT 합산

다음 실행은 평균 PCC 20 MW와 PUE 1.15를 향후 완성 AIDC의 목표로 두고,
그 직전 단계인 평균 약 17.39 MW의 IT 부하를 node 단위로 합산합니다.

```powershell
python scripts\run_nlr_bottom_up_facility.py `
  --target-pcc-mean-mw 20 `
  --pue 1.15 `
  --target-utilization 0.8 `
  --duration-min 15 `
  --output-dir outputs\nlr_bottom_up_facility_17mw_it
```

구조는 `[3]`의 request 분배, model-instance 수 결정, node pool 할당,
incomplete request 및 idle node 집계를 따릅니다. 단, 논문의 empirical
`P(request-rate)` 할당 블록을 본 연구의 M3+R3 node generator로 교체했습니다.
PUE는 평균 IT 목표를 계산할 때만 사용하며 순간 IT 전력에 곱하지 않습니다.

## 현재 배전 테스트베드: IEEE 33-bus, 12.66 kV

사용자 선택에 따라 원형 전압 12.66 kV를 유지합니다. 현재 설정은
`configs/ieee33_1266kv_screen_v1.json`, 전체 경로 등록은
`configs/grid_testbed_registry.json`에 있습니다. 앞선 33 kV 후보와 결과는
비교 이력으로 보존하며 현재 선택 조건으로 사용하지 않습니다.

```powershell
.\.venv-grid\Scripts\python.exe -B scripts\screen_ieee33_1266kv_v1.py
.\.venv-grid\Scripts\python.exe -B -m pytest tests\test_ieee33_1266kv_v1.py -q --junitxml=outputs\ieee33_1266kv_screen_v1\test_results.xml
.\.venv-grid\Scripts\python.exe -B scripts\report_ieee33_1266kv_v1.py
```

기존 AIDC/ESS/TES를 변경하지 않은 1,438개 5분 운전점 검토입니다.
원형 배경부하부터 0.95 p.u. 연구 하한에 미달하고, 20 MW대 AIDC는 기존
합성 선로 한계 600 A를 초과합니다. 기준운전 부적합 시 정적 수용량은
null이며 0 MW나 DOE로 해석하지 않습니다. 기존 공동운전 계획 재생은
계통 기반 재최적화가 아닙니다. 이후 부하 규모·공급 구조·기준 운전 조건의
정합성을 결정해야 합니다. 보고서는 `outputs/ieee33_1266kv_screen_v1/report_ko.html`입니다.

## Git 운용 원칙

- 원본 NLR 데이터, 가상환경, 모델 checkpoint 및 자동 생성 결과는 커밋하지 않습니다.
- 코드, 설정 예시, 작은 검증용 샘플 및 재현 가능한 분석 절차만 버전 관리합니다.
- 기능 단위 브랜치와 명확한 commit message를 사용합니다.

첫 개발 브랜치 예시:

```powershell
git switch -c feature/nlr-feature-extraction
```
