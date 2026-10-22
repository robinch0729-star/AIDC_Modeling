# 글로벌 과제 진행 - NLR gray-box 모델링 추가 진행사항

> 본 문서는 기존 보고 내용인 `NLR 데이터셋 구조·metadata 탐색 → workload 특성 추출 → condition audit → gray-box 수식 선정 진행` 이후의 추가 수행 내용을 정리한 것이다.

> **현재 도달점:** Prompt-conditioned M3 정상상태 모델과 R2 OU/R3 Jump-OU residual 후보를 구성하고, 두 후보를 병렬 적용한 20 MW IT workload-group aggregation v2까지 완료하였다. R2는 통계적 baseline일 뿐 최종모델이 아니며, R2/R3 최종 선택은 후속 DOE violation·decision regret 평가에서 수행한다.

### 인용표기 안내

- `[PL#]`: 현재 `paper_list`에 포함된 문헌
- `[EXT#]`: `paper_list` 외부의 peer-reviewed 문헌
- `본 연구 분석`: 해당 문헌의 수식이나 결과를 그대로 사용한 것이 아니라 NLR 데이터에 대해 직접 구성·계산한 내용
- 문헌을 인용했더라도 동일한 수식을 그대로 사용하지 않은 경우에는 `방법론적 근거` 또는 `응용·변형`으로 별도 표시

---

## 0. 이전 연구 진행상황

### 0.1. 연구목표 및 데이터 활용방향 설정

- 전체 연구방향
  - NLR workload 측정자료를 이용하여 AIDC의 IT 부하특성을 분석
  - 측정자료에서 정상상태 전력과 동적 변동특성을 분리
  - 물리적으로 해석 가능한 수식구조와 data-driven 보정을 결합한 gray-box IT 부하모델 구성
  - 향후 MW급 AIDC profile 생성 및 배전계통 영향평가에 활용

- 초기 데이터 처리 flow

$$
\text{NLR 원본파일 탐색}
\rightarrow
\text{schema·metadata 정리}
\rightarrow
\text{신호 품질검증}
\rightarrow
\text{workload 특성추출}
\rightarrow
\text{condition 분류·audit}
\rightarrow
\text{gray-box 수식 후보 선정}
$$

- 근거 문헌
  - Paper list [3]은 NLR Kestrel에서 H100 GPU 기반 AI workload 전력을 0.1초 해상도로 측정하고, 이를 facility-level profile로 확장하는 연구방향과 데이터 구조를 제공함 `[PL3]`
  - 데이터센터 부하를 구성요소 및 계층별로 분해하여 모델링하는 일반적 접근은 데이터센터 전력모델 survey와 simplified IT model을 참고함 `[PL9, EXT1]`

---

### 0.2. NLR 데이터 catalog 및 schema 구축

- NLR dataset 전체 파일을 무작위로 바로 학습에 사용하지 않고, 우선 파일 단위 catalog를 생성
- 폴더·파일별로 다음 정보를 정리
  - 파일경로 및 파일형식
  - Workload family
  - Prompt dataset
  - Request rate
  - Maximum output tokens
  - 계측신호 종류 및 column schema
  - 시작·종료시각과 profile 길이
  - Sampling interval 및 데이터 개수

- 서로 다른 파일형식을 공통 분석구조로 읽기 위한 코드 작성
  - `src/nlr_data/catalog.py`
  - `src/nlr_data/schema_registry.py`
  - `src/nlr_data/readers.py`
  - `notebooks/01_nlr_data_overview.ipynb`

- 대표 파일을 실제로 읽어 time, CPU/GPU/aggregated IT power 등 주요 column이 정상적으로 변환되는지 검증

- 주요 산출물
  - `nlr_file_catalog.csv/.parquet`
  - `schema_registry.csv`
  - `representative_reader_validation.csv`

- 문헌과의 관계
  - 원본 workload 유형과 측정환경은 Paper list [3]을 기준으로 해석함 `[PL3]`
  - Catalog·schema registry·reader 구현은 공개 NLR 파일을 안정적으로 분석하기 위해 **본 연구에서 수행한 데이터 엔지니어링 작업**임

---

### 0.3. 시간축 및 신호 품질검증

- 각 profile을 모델보정에 사용하기 전에 다음 항목을 검사
  - 시간축 단조 증가 여부
  - 중복 timestamp
  - 결측치 및 비수치 값
  - Sampling interval의 규칙성
  - 비정상적으로 짧은 profile
  - 음수·비정상 전력값
  - 분석 가능한 active-power channel 존재 여부

- 분석정책
  - 원본파일은 변경하지 않고 분석용 profile에서만 정렬·중복제거·유효구간 선별
  - 원본 sampling interval과 보간 여부를 별도 기록
  - 품질검증을 통과하지 못한 profile은 exclusion reason과 함께 저장

- 구현코드
  - `src/nlr_data/signal_validation.py`
  - `notebooks/02_nlr_signal_validation.ipynb`

- 주요 산출물
  - `profile_quality_report.csv/.parquet`
  - `run_manifest.csv/.parquet`
  - `extraction_failures.csv`

- 문헌과의 관계
  - Paper list [3]에서 제시한 0.1초 workload power measurement를 분석대상으로 사용함 `[PL3]`
  - 세부 품질판정 규칙과 분석정책은 후속 parameter calibration에서 오류가 전파되지 않도록 **본 연구에서 정의**함

---

### 0.4. NLR workload 유형 분류

- Dataset에 포함된 workload를 다음과 같이 분류
  - Training/Fine-tuning
  - Inference
    - Offline inference
    - Online inference finite
    - Online inference rate
  - Whole-facility profile
  - Colocation profile

| Workload | 요청 전달방식 | 종료조건 | 주요 활용목적 |
|---|---|---|---|
| Offline inference | 여러 prompt를 한 번에 전달 | 모든 prompt 처리 완료 | 처리량·job 전체 전력특성 분석 |
| Online inference finite | 정해진 수의 요청을 일정 rate로 전달 | 정해진 요청 수 처리 완료 | 기동·종료 및 유한 workload 분석 |
| Online inference rate | 일정 request rate를 지속적으로 유지 | 설정시간 경과 | 정상상태 request-rate–전력 관계 분석 |

- 현재 gray-box 정상상태 분석대상은 **online inference rate의 aggregated IT power**로 결정
- 선정 이유
  - Request rate가 명확한 독립변수로 주어짐
  - 동일 조건의 sustained profile을 이용하여 condition 평균전력 계산 가능
  - 저율부터 포화영역까지 request-rate–power 관계 비교 가능

- 근거 문헌
  - Workload 분류와 각 실험의 원래 목적은 Paper list [3]의 NLR 실험설계를 따름 `[PL3]`
  - `online inference rate`만 우선 정상상태 gray-box 분석대상으로 한 것은 **본 연구 범위설정**임

---

### 0.5. Workload profile 특성추출

- 각 유효 profile에서 AIDC workload의 크기·변동성·ramp·event를 설명할 수 있는 feature를 추출

- 주요 추출항목
  - 평균 power
  - 최대·최소 power
  - 표준편차 및 coefficient of variation
  - Peak-to-average ratio
  - 평균·최대·95th-percentile absolute ramp
  - Burst/dip 발생횟수
  - Burst/dip peak magnitude
  - Burst/dip 지속시간
  - Sampling interval 및 profile duration

- 구현코드
  - `src/nlr_data/feature_extraction.py`
  - `src/nlr_data/feature_analysis.py`
  - `notebooks/03_nlr_feature_analysis.ipynb`

- 주요 산출물
  - `workload_features.csv/.parquet`
  - `condition_uncertainty_features.csv/.parquet`
  - `workload_feature_summary.csv/.parquet`
  - `graybox_target_candidates.csv`
  - `dfl_information_candidates.csv`

- 근거 문헌
  - Paper list [3]은 workload별 평균·최대전력, transient, ramp 및 facility-level utilization profile을 주요 분석대상으로 사용함 `[PL3]`
  - Paper list [2]는 stochastic data-center load model이 단순 평균오차뿐 아니라 시간적 pattern을 재현해야 한다는 근거를 제공함 `[PL2]`
  - 구체적인 feature 정의와 threshold는 NLR profile을 gray-box·DFL 입력정보로 변환하기 위해 **본 연구에서 구성**함

---

### 0.6. Condition ID 생성 및 experiment-design audit

- `condition_id`의 의미
  - NLR 원본파일에 하나의 완성된 `condition_id` column이 제공된 것이 아님
  - 원본 metadata에 존재하는 workload 조건을 조합하여 **본 연구에서 생성한 실험조건 식별자**임

- Condition ID가 필요한 이유
  - 동일 실험조건의 여러 파일을 하나의 그룹으로 묶기 위함
  - 서로 다른 실험조건을 구분하기 위함
  - Condition별 평균전력·변동성·불확실성 계산
  - Train/validation/test 간 동일조건 중복 방지

- Workload별 condition 구성변수

| NLR workload | Condition을 구분하는 주요 변수 |
|---|---|
| Online inference rate | Prompt dataset + Request rate |
| Online inference finite | Prompt dataset + Request rate + Maximum output tokens |
| Offline inference | Prompt 개수 + Maximum output tokens |

- Audit 전 초기 grouping
  - Offline inference: 80 conditions
  - Online inference finite: 57 conditions
  - Online inference rate: 100 conditions
  - Training/Fine-tuning: 8 conditions
  - Whole-facility inference: 4 conditions
  - Colocation: 4 conditions
  - 총 253 conditions
  - 이 값은 기존 진행보고에 사용된 초기 condition 수임

- Experiment-design audit 후 수정

| Workload | Audit 전 condition 수 | Audit 후 권장 condition 수 | 판단 |
|---|---:|---:|---|
| Offline inference | 80 | 80 | 기존 key 유지 가능 |
| Online inference finite | 57 | 114 | Prompt dataset을 별도 controlled factor로 추가 |
| Online inference rate | 100 | 200 | Prompt dataset 2종을 반복실험으로 합치지 않고 분리 |
| Training/Fine-tuning | 8 | 8 | 로컬 release 기준 유지 |
| Whole-facility inference | 4 | 4 | Simulation scenario로 별도 관리 |
| Colocation | 4 | 4 | Simulation scenario로 별도 관리 |
| **합계** | **253** | **410** | Audit 후 registry 갱신 |

- 추가 확인사항
  - Paper list [3]에는 training/fine-tuning nominal condition이 9개로 제시되지만, 로컬 aggregated metadata에는 1-node Stable Diffusion case가 없어 8개로 확인됨
  - Online inference finite와 online inference rate의 초기 ID는 두 prompt dataset을 동일 condition의 반복값처럼 합쳤음
  - 실제 실험설계에서는 prompt dataset이 controlled factor이므로 각각 114개와 200개 condition으로 분리하는 것이 타당함

- Audit에서 확인한 사항
  - 동일 condition 내부 metadata 일관성
  - 서로 다른 condition이 같은 ID로 합쳐지는 collision 여부
  - Paper list [3]의 표·본문에 제시된 실험설계와 실제 파일구조의 일치 여부
  - Prompt dataset이 독립적인 전력특성 차이를 만드는 조건인지 여부

- 구현코드
  - `src/nlr_data/condition_audit.py`
  - `src/nlr_data/condition_targets.py`
  - `notebooks/04_nlr_condition_audit.ipynb`

- 주요 산출물
  - `condition_registry.csv/.parquet`
  - `experiment_design_registry.csv/.parquet`
  - `condition_validation_summary.csv/.parquet`
  - `modeling_cohort_manifest.csv/.parquet`
  - `prompt_dataset_effect_report.csv/.parquet`

- 근거 문헌
  - Condition을 구성하는 원본 metadata와 workload별 실험설계는 Paper list [3]을 따름 `[PL3]`
  - 구조화된 표본을 condition/block 단위로 관리하고 분할하는 원칙은 structured cross-validation 문헌을 참고함 `[EXT2]`
  - Condition ID 문자열과 audit 절차 자체는 **본 연구에서 설계·구현**함

---

### 0.7. 이전 보고 시점의 도달단계

- 완료된 작업
  - NLR 파일 catalog 및 schema registry 구축
  - 00–03 계열 workload 파일의 reader 구현
  - 시간축·sampling·결측·중복·전력신호 품질검증
  - Profile별 workload feature 추출
  - Workload family 및 condition 구조 정리
  - Condition ID 생성과 실제 NLR experiment-design audit
  - Gray-box 분석대상을 `online inference rate aggregated IT power`로 선정

- 당시 진행 중이었던 작업
  - Request rate와 condition 평균 IT 전력 관계를 나타낼 정상상태 수식 후보 구성
  - 수식 후보별 train/validation/test 비교 준비

- 이후 추가 수행내용
  - 아래 `1. Online inference rate 정상상태 부하식 비교`부터 M1·M2·M3 비교, residual 동특성 모델링 및 MW급 집합화 예비실험을 진행

---

## 1. Online inference rate 정상상태 부하식 비교

- 분석 대상
  - NLR `online inference rate` workload의 aggregated IT power
  - Request rate: 1–100 request/s
  - Prompt dataset: `InstructCoder`, `mlperf-inference-llama2-data`
  - 총 200개 condition

- 목적
  - Request rate 증가에 따른 평균 IT 전력의 정상상태 관계를 나타내는 수식 선정
  - 단순 fitting 성능뿐만 아니라 단조성 및 포화특성 확인

- 근거 문헌
  - **데이터 근거:** Paper list [3]은 NLR Kestrel의 H100 GPU 기반 training·fine-tuning·inference workload 전력을 0.1초 해상도로 측정하고, node-level profile을 facility level로 확장하는 절차를 제시함 `[PL3]`
  - **IT 부하 수식 후보를 비교하는 근거:** 데이터센터 IT 전력모델은 utilization 또는 workload와 전력의 관계를 선형·비선형·구성요소 기반으로 표현할 수 있으며, 하나의 수식이 모든 설비에 일반화되지 않는다는 점이 기존 survey와 simplified IT model에서 지적됨 `[PL9, EXT1, EXT5]`
  - **M3 logistic 형태의 직접적 선행 예:** Paper list [17]은 GPU batch size와 GPU 전력의 포화관계를 logistic 함수로 fitting함. 다만 본 연구는 입력을 batch size가 아닌 request rate로 변경했으므로 `응용·변형`에 해당하며, [17]은 preprint임 `[PL17]`

- 비교한 정상상태 수식

### M1: Linear model

$$
P(r)=P_0+\alpha r
$$

- 가장 단순한 baseline
- 고 request-rate에서도 전력이 계속 증가하므로 포화영역을 표현하기 어려움

### M2: Cubic model

$$
P(r)=b_0+b_1u+b_2u^2+b_3u^3,
\qquad
u=\frac{r-r_{\min}}{r_{\max}-r_{\min}}
$$

- 비선형 곡률을 유연하게 표현
- 학습범위 밖에서 전력이 감소하거나 재상승하는 비물리적 거동이 발생할 수 있음

### M3: Logistic saturation model

$$
P_{\mathrm{ss}}(r,d)
=P_0(d)+
\frac{\Delta P(d)}
{1+\exp\left[-k(d)\{\ln(1+r)-x_0(d)\}\right]}
$$

- $r$: request rate
- $d$: prompt dataset
- $P_0$: 저 request-rate 구간의 base IT power
- $P_0+\Delta P$: 포화 IT power
- $x_0$: 중간 전력에 도달하는 request-rate 위치
- $k$: 포화영역으로 전환되는 기울기

---

## 2. Condition-aware 데이터 분할

- 동일 request rate의 두 prompt dataset이 서로 다른 split으로 나뉘지 않도록 구성
- 동일 운전조건이 train과 validation/test에 동시에 포함되는 data leakage 방지

- 근거 문헌
  - NLR의 condition 구조와 두 prompt dataset의 실험조건은 Paper list [3]을 따름 `[PL3]`
  - 구조화된 데이터에서는 관련 표본을 무작위로 흩어 놓기보다 group/block 단위로 분리해야 과도하게 낙관적인 validation 결과를 줄일 수 있다는 cross-validation 원칙을 적용함 `[EXT2]`
  - 본 연구에서는 이 원칙을 `동일 request rate의 두 prompt dataset을 같은 split에 배치`하는 방식으로 구현함

| 구분 | Request-rate 구성 | Condition 수 | 목적 |
|---|---:|---:|---|
| Train | 저·중 request-rate 65개 × dataset 2종 | 130 | 모델 파라미터 추정 |
| Validation | Train 범위 내부의 15개 rate × dataset 2종 | 30 | 모델 family 선정 |
| Test | 81–100 request/s × dataset 2종 | 40 | 고율 외삽 성능 확인 |

---

## 3. 정상상태 모델 비교 결과

- 결과의 성격
  - 아래 수치는 외부 문헌에서 가져온 값이 아니라 NLR condition dataset에 대해 직접 수행한 **본 연구 분석결과**임
  - 평가대상 데이터의 출처는 Paper list [3], 후보식 구성의 일반적 근거는 `[PL9, EXT1, EXT5]`임

| 모델 | Validation RMSE | Validation MAPE | Validation $R^2$ | Test RMSE | Test MAPE | 단조성 위반 |
|---|---:|---:|---:|---:|---:|---:|
| M1 linear | 0.14775 kW | 4.825% | 0.6388 | 0.31012 kW | 10.925% | 0% |
| M2 cubic | 0.04328 kW | 1.269% | 0.9690 | 0.11865 kW | 3.676% | 32.3–36.1% |
| **M3 logistic** | **0.03470 kW** | **1.059%** | **0.9801** | **0.04243 kW** | **1.422%** | **0%** |

- 최종 선정: **M3 logistic saturation model**
- 선정 이유
  - Validation 및 high-rate test 오차가 가장 작음
  - Request rate 증가에 따른 단조 증가 보장
  - 유한한 포화전력 표현 가능
  - M2에서 발생한 고율구간의 비물리적 곡률을 방지

![M1-M3 정상상태 모델 비교](figures/model_comparison_prompt_conditioned.png)

> Figure. Prompt dataset별 M1·M2·M3 fitting 결과. 파란 점은 NLR condition 평균전력, 녹색선은 선정된 M3 모델을 의미한다.

---

## 4. Prompt dataset별 M3 파라미터 추정

| Prompt dataset | $P_0$ | $\Delta P$ | 포화전력 $P_0+\Delta P$ | $k$ | $x_0$ | 환산 half-rate |
|---|---:|---:|---:|---:|---:|---:|
| InstructCoder | 2.11876 kW | 0.73334 kW | 2.85210 kW | 4.1390 | 2.8745 | 약 16.7 request/s |
| mlperf-inference-llama2-data | 2.11560 kW | 0.67918 kW | 2.79478 kW | 4.4851 | 3.0479 | 약 20.1 request/s |

- 동일 request rate에서도 prompt dataset에 따라 소비전력이 달라짐
- Prompt dataset별 파라미터를 허용했을 때 pooled M3보다 RMSE 감소
  - Validation: 약 36% 감소
  - Test: 약 17% 감소
- 따라서 prompt dataset은 단순 반복조건이 아니라 정상상태 부하 파라미터를 변화시키는 workload 조건으로 유지

- 근거 문헌
  - Paper list [3]은 inference 전력이 request rate뿐 아니라 prompt dataset, prompt 수, maximum output tokens 등의 workload 조건에 따라 달라지는 실험구조를 제공함 `[PL3]`
  - Prompt-dataset-conditioned parameter 자체와 표의 추정값은 **본 연구에서 NLR 데이터로 보정한 결과**이며 [3]이 동일한 M3식을 제안한 것은 아님

> 주의: M3는 현재 데이터에서 후보 수식을 비교하여 선정한 empirical gray-box backbone이다. 기존 문헌의 수식을 그대로 사용한 것은 아니므로, peer-reviewed 문헌에 기반한 최종 수식 근거는 추가 검토가 필요하다.

---

## 5. 정상상태 모델로 설명되지 않는 동특성 분리

- 실제 profile과 M3 정상상태 전력의 차이를 residual로 정의

$$
e(t\mid r,d)
=P_{\mathrm{meas}}(t\mid r,d)-P_{\mathrm{ss}}(r,d)
$$

- Residual에서 확인할 특성
  - 변동성 및 분포
  - Autocorrelation function, ACF
  - Ramp 분포
  - Power spectral density, PSD
  - 순간적인 burst/dip event의 크기와 지속시간

- 1차 지연 모델 적용 가능성 검토

$$
\tau\frac{dP(t)}{dt}=P_{\mathrm{ss}}(t)-P(t)
$$

- 검토 결과
  - NLR `online inference rate` profile은 고정 request rate를 유지하는 실험임
  - 입력 request rate의 명확한 step 변화가 기록되지 않아 causal plant time constant $\tau$를 직접 식별하기 어려움
  - 출력 profile의 자기상관시간을 물리적 입력–출력 시정수로 해석하지 않음
  - 따라서 현 단계에서는 1차 지연보다 residual의 통계·시간 특성 재현에 집중

- 근거 문헌
  - NLR 측정결과는 AI training과 inference profile에 빠른 전력변동 및 transient가 존재함을 보여주므로, condition 평균값만으로 전체 profile을 표현하기 어렵다는 데이터 근거를 제공함 `[PL3]`
  - Paper list [2]는 확률적 데이터센터 부하의 point-wise 오차보다 pattern consistency를 이용해 물리모델을 보정해야 한다는 방향을 제시함 `[PL2]`
  - Paper list [11]은 grid-level 데이터센터 해석에서 workload fluctuation, 전력변환기 및 보호 동특성을 정적 ZIP/PQ 모델만으로 표현하기 어렵다는 점을 지적함 `[PL11]`
  - Residual 정의 $e=P_{\mathrm{meas}}-P_{\mathrm{ss}}$와 현재의 식별 가능성 판단은 **본 연구 분석**임

---

## 6. Residual 구조 진단 및 후보모델 수정

- M3로 condition 평균 운전점을 제거한 residual에 대해 **train split 130개 profile만** 사용하여 구조를 먼저 진단
- 새 절차에서는 AR(1)/OU를 적합한 뒤 innovation의 자기상관과 분포를 진단하고, Gaussian 대비 3-component GMM의 BIC 개선 및 tail ratio로 jump 구조의 필요성을 판단

### 6.1. 후보모델

| 모델 | 현재 구조 | 역할 및 문헌 근거 |
|---|---|---|
| R0 | M3 정상상태값만 사용 | 동특성이 없는 baseline |
| R1 | Gaussian white noise | 분산만 재현하는 확률 baseline |
| R2 | OU의 정확 이산형인 AR(1) | 단기 자기상관 재현 `[PL38]` |
| R3 | Signed Jump-OU의 이산시간 transition-mixture 근사 | 자기상관과 heavy-tail burst/dip 재현 `[PL2, PL38]` |
| R4 | Train residual empirical replay | 비모수 성능상한 reference; deployable model 선정대상 제외 |

### R2: OU/AR(1)

$$
de(t)=-\theta e(t)dt+\sigma dW(t)
$$

$$
e_{k+1}=\phi e_k+\eta_k,
\qquad
\phi=e^{-\theta\Delta t},
\qquad
\eta_k\sim\mathcal N(0,\sigma_\eta^2)
$$

- 연속시간 OU 모델을 sampling interval $\Delta t$에서 이산화하면 AR(1) 형태가 됨
- 예전 R2와 모델 family는 같고, 현재는 OU–AR(1) 관계와 parameter 의미를 명시한 것임

### R3: Signed Jump-OU

$$
de(t)=-\theta e(t)dt+\sigma dW(t)+dJ(t)
$$

$$
e_{k+1}=\phi e_k+\xi_k,
\qquad
\xi_k\sim
\sum_{c\in\{-,0,+\}}\pi_c\mathcal N(\mu_c,\sigma_c^2)
$$

- $-$ component: 급격한 power dip
- $0$ component: 평상시 background innovation
- $+$ component: 급격한 power burst
- 3-component GMM은 Jump-OU의 small-$\Delta t$ transition을 NLR 데이터에 보정하기 위한 **본 연구의 이산 근사·calibration 방법**임
- Train condition에서 Gaussian 대비 GMM BIC 개선이 충분하지 않으면 jump gate가 0에 가까워져 R2 구조로 수축하도록 설계

### 6.2. Train-only 구조진단 결과

| 진단항목 | 결과 | 해석 |
|---|---:|---|
| Median $\phi$ | 0.9709 | residual에 강한 단기 지속성 존재 |
| Residual serial-correlation 검출비율 | 100% | R1 white noise만으로 부족 |
| GMM BIC 개선 profile 비율 | 73.08% | 단일 Gaussian보다 mixture가 유리한 condition 다수 |
| Jump-evidence profile 비율 | 72.31% | R3를 검증후보로 유지할 근거 |
| Median Gaussian 3-$\sigma$ tail ratio | 3.706 | Gaussian 가정보다 tail 발생이 많음 |
| Innovation이 white하지 않은 비율 | 100% | R2 역시 완전한 동특성모델이라고 단정 불가 |
| Mean-shift evidence 비율 | 11.54% | 일부 condition은 비정상성 추가 검토 필요 |

- 결론: R1은 baseline으로만 유지하고, R2와 R3를 validation에서 비교
- 근거 문헌
  - `[PL38]`은 데이터센터 IT 부하 residual에 OU process를 사용하고 workload jump/pulse를 별도 반영하는 peer-reviewed 근거를 제공함
  - `[PL2]`는 mean-reverting stochastic process와 Poisson jump를 포함하는 데이터센터 동적 부하 calibration 구조를 제시함. 단, preprint이므로 R3의 보조근거로 사용함
  - `[PL37]`은 AR(1) residual과 time-domain·PSD·분포 검증 필요성을 뒷받침함. 단, preprint임
  - GMM/BIC gate와 수치 결과는 **본 연구의 NLR train-only 분석**임

---

## 7. Residual 모델 평가척도

- 시간점별 waveform RMSE만으로 판단하지 않고 실제 NLR residual의 통계·시간 특성 재현 여부를 종합 평가

$$
J=
0.15E_{\mathrm{var}}
+0.15E_{\mathrm{dist}}
+0.20E_{\mathrm{ACF}}
+0.15E_{\mathrm{ramp}}
+0.10E_{\mathrm{PSD}}
+0.15E_{\mathrm{tail}}
+0.10E_{\mathrm{innovation\ ACF}}
$$

- $J$가 낮을수록 실제 residual 특성을 잘 재현
- 평가항목
  - Variability: 표준편차·변동계수 재현오차
  - Distribution: residual 분위수·분포 재현오차
  - ACF: 시간 자기상관 재현오차
  - Ramp: residual ramp 분포 재현오차
  - PSD: 저주파 전력비율 재현오차
  - Innovation tail: Gaussian tail ratio와 excess kurtosis 재현오차
  - Innovation ACF: OU 제거 후 innovation의 남은 시간상관 재현오차
- Profile/model 조합별 Monte Carlo simulation 수행
- R4 empirical replay는 실제 donor trace가 필요한 reference이므로 deployable model 선정 대상에서 제외
- 고정 가중치 결과에만 의존하지 않도록 7개 오차항의 가중치를 Dirichlet 분포에서 100,000회 무작위 추출하여 순위 민감도 분석

- 근거 문헌
  - Paper list [2]는 stochastic load model을 point-wise RMSE만으로 평가하지 않고 패턴 수준에서 비교해야 한다는 근거를 제공함 `[PL2]`
  - Lee and Baldick은 전력 시나리오를 statistical, spectral-density 및 correlation analysis로 검증하여, 생성 profile의 분포·주파수·상관특성을 함께 평가하는 선행 근거를 제공함 `[EXT3]`
  - Paper list [37] 역시 aggregate AI workload trace를 time-domain error, PSD 및 marginal distribution으로 비교함 `[PL37]`
  - 위 $J$의 세부항목과 가중치는 **본 연구의 모델선정용 종합지표**이며, 문헌 원식을 그대로 가져온 것은 아님

---

## 8. Residual 모델 선정 결과

- 아래 score는 외부 문헌의 값이 아니라 NLR validation profile 30개에 대한 **본 연구 Monte Carlo 분석결과**임

| 모델 | Validation mean score | Validation median score | 현재 판단 |
|---|---:|---:|---|
| R4 empirical replay | 0.3534 | 0.2143 | 비모수 reference |
| **R2 OU** | **0.4796** | 0.5134 | 평균 score 기준 통계 baseline |
| R3 Jump-OU | 0.5059 | **0.3872** | tail/ramp-aware 후보 유지 |
| R1 white noise | 0.7775 | 0.7192 | 시간상관·ramp 재현 부족 |
| R0 M3 only | 5.1858 | 5.1345 | 동특성 재현 불가 |

- 평균 validation score만 보면 R2가 가장 낮아 **provisional statistical baseline**으로 선정
- 그러나 R2–R3 paired score 차이의 bootstrap 95% CI는 $[-0.0838,\ 0.1550]$으로 0을 포함
- Condition별 비교에서는 R2의 win fraction이 36.67%, 즉 R3가 약 63.33% condition에서 더 낮은 score를 보였음
- 가중치 100,000회 민감도 분석
  - R2 선택비율: 68.82%
  - R3 선택비율: 30.96%
  - R1 선택비율: 0.23%
- 따라서 **R2를 최종 모델로 확정하지 않고 R2·R3를 병렬 유지**
- 최종 family는 downstream DOE violation, constraint violation 및 decision regret 차이를 포함하여 결정

![Residual 모델 validation score](figures/validation_model_scores_v2.png)

> Figure. Residual 모델별 validation 통계재현 score. 낮을수록 실제 NLR residual 특성에 가까우며 회색 R4는 reference-only 모델이다.

![R2 high-rate test 대표 profile](figures/representative_high_rate_test_v2.png)

> Figure. Validation에서 평균 score가 가장 낮았던 R2를 train+validation에서 재보정한 뒤 81–100 request/s test 조건에 적용한 대표 결과. 이 test 결과는 R2의 외삽 확인이며 R2–R3 최종 DOE 비교를 대신하지 않는다.

---

## 9. 현재 구성된 equivalent workload-group IT 부하 생성기

- Prompt-dataset-conditioned M3 정상상태와 R2 또는 R3 residual을 결합

$$
P_{\mathrm{IT}}(t\mid r,d)
=P_{\mathrm{ss}}^{\mathrm{M3}}(r,d)
+e_m(t\mid r,d),
\qquad m\in\{\mathrm{R2},\mathrm{R3}\}
$$

- 입력정보
  - Prompt dataset
  - Request rate
  - Profile 길이 및 sampling interval
  - Random seed

- 출력정보
  - 평균 IT 전력
  - 시간상관을 갖는 stochastic fluctuation
  - R3 사용 시 heavy-tail burst/dip innovation
  - Ramp·분포·ACF·PSD 특성을 포함한 time-series profile

- 재사용 가능한 parameter artifact 생성
  - M3 parameter
  - Condition별 OU/innovation statistics
  - R3 negative/background/positive mixture parameter
  - Train residual library는 R4 reference용으로만 유지

- 해석상 주의
  - NLR profile 하나가 정확히 몇 대의 서버 또는 GPU를 대표하는지 확정되지 않았음
  - 따라서 현재 생성 단위를 물리적 `server/job`이 아니라 **equivalent workload group**으로 부름
  - R2와 R3 중 하나를 최종 채택한 상태가 아님

- 근거 문헌
  - Paper list [3]의 measured workload profile과 bottom-up facility modeling 방향을 데이터 기반으로 사용함 `[PL3]`
  - Paper list [2]의 `물리적 backbone + pattern-aware data calibration` 개념을 gray-box 구성의 방법론적 근거로 사용함 `[PL2]`
  - Paper list [15]와 [16]은 각각 thermal dynamics 및 total data-center load 분야에서 물리구조와 data-driven correction을 결합하는 hybrid/gray-box 접근의 peer-reviewed 사례를 제공함 `[PL15, PL16]`
  - `[PL38]`의 OU 기반 residual 구조와 `[PL2]`의 jump-process 구조를 후보 family의 직접 근거로 사용함
  - 현재의 `M3+R2/R3` 결합, condition-neighbour parameter interpolation 및 generator 구현은 **본 연구의 NLR 적용·구현**임

---

## (구) 10. 20 MW 집합화 예비 실험: 최종 방법론에서 제외

- 아래 내용은 workload 다양성과 동기성이 변동성·ramp에 미치는 영향을 탐색했던 **이전 버전 기록**임
- 현재 결과나 최종 모델로 사용하지 않으며, 신규 결과는 `10. R2/R3 병렬 workload-group aggregation v2`에 정리

- 예비 실험조건
  - 평균 IT 전력: 20 MW
  - 분석시간: 180 s
  - 해상도: 0.1 s
  - Scenario당 Monte Carlo 10회
  - 대표 조건: 90 request/s

- 당시 구현한 예비 common-factor aggregation 식

$$
e_{\mathrm{agg}}(t)
=\sqrt{\rho}\,e_{\mathrm{common}}(t)
+\sqrt{1-\rho}\,
\frac{1}{N_{\mathrm{eff}}}
\sum_{i=1}^{N_{\mathrm{eff}}}e_i(t)
$$

$$
\frac{\sigma_{\mathrm{agg}}}{\sigma_{\mathrm{unit}}}
\approx
\sqrt{\rho+\frac{1-\rho}{N_{\mathrm{eff}}}}
$$

- $N_{\mathrm{eff}}$: 비동기 workload 다양성을 나타내기 위해 가정한 통계적 유효 unit 수
- $\rho$: 여러 unit이 함께 변하는 common-mode 비율로 가정한 민감도 변수

- 근거 문헌
  - Paper list [3]은 node-level AI workload profile을 job arrival·scheduling·utilization과 결합하여 facility-level time series로 확장하는 bottom-up 방법을 제시함 `[PL3]`
  - Fan et al.은 수백~수천 서버의 동시 peak가 단순 정격합과 다르며 aggregation level에 따라 statistical diversity가 나타남을 실측함 `[EXT5]`
  - Sevlian and Rajagopal은 부하 집합화가 변동성과 상대 예측오차를 감소시키지만 일정 규모 이후에는 개선이 포화될 수 있음을 대규모 데이터로 분석함 `[EXT4]`
  - Lee and Baldick은 correlated load scenario가 uncorrelated scenario와 다른 계통계획 결과를 만들 수 있으므로 공통·개별 변동과 상관구조를 보존해야 한다고 제시함 `[EXT3]`
  - Paper list [17]은 여러 GPU replica profile을 단순히 하나의 trace로 확대하지 않고, 서로 다른 시작시점을 적용하여 중첩하는 집합 profile 생성 예를 제시함. 단, preprint이며 request-rate 집합화식 자체를 제공하는 것은 아님 `[PL17]`
  - 위 $N_{\mathrm{eff}}$·$\rho$ 수식은 이러한 문헌의 문제의식을 바탕으로 만든 **본 연구 예비 민감도식**이며 문헌 원식이 아님

![구버전 20 MW 집합부하 예비 profile](figures/representative_20mw_profiles.png)

> Figure. 구버전 민감도 결과. 출처가 명확하지 않은 $\rho$ 수식을 사용했으므로 최종 결과에서 제외한다.

![집합부하 변동성 민감도](figures/diversity_cv_curves.png)

> Figure. 독립 unit 수와 common-mode 비율에 따른 coefficient of variation 변화. 독립 변동은 unit 수 증가에 따라 평균화되지만 common-mode 성분은 남는다.

![집합부하 ramp 민감도](figures/diversity_ramp_curves.png)

> Figure. 독립 unit 수와 common-mode 비율에 따른 95th-percentile absolute ramp 변화.

---

## (구) 11. 집합화 방법론의 폐기 결정

- 위 20 MW 결과는 **집합화 개념을 확인하기 위한 예비 민감도 실험**임
- 현재 NLR 데이터만으로는 다음 값을 직접 식별할 수 없음
  - 실제 물리 서버 또는 workload group 수
  - 통계적 $N_{\mathrm{eff}}$
  - Facility common-mode 비율 $\rho$
  - 여러 workload가 동시에 실행될 때의 cross-correlation

- 기존 common-factor decomposition 참고문헌은 preprint이고 접근성에도 문제가 있어 핵심 방법론의 근거에서 제외
- 위 $\sqrt{\rho}$ aggregation 식은 특정 문헌의 원식을 그대로 적용한 것이 아니므로 **현재 코드와 최종 부하모델에서 제외**
- 기존 `src/nlr_data/workload_generator.py` 및 `notebooks/09_nlr_workload_aggregation.ipynb`는 재현성 보존을 위한 legacy 파일이며, 보고·논문 결과에는 사용하지 않음

- 관련 문헌과 재검토 근거
  - `[EXT3]`은 common/idiosyncratic decomposition의 정식 peer-reviewed 전력계통 사례지만, 본 연구에서 사용한 $\sqrt{\rho}$·$N_{\mathrm{eff}}$ 식과 동일하지 않음
  - `[EXT4]`는 aggregation smoothing의 일반적 근거이지만 AIDC 내부 workload 동시성이나 물리적 server count를 식별해 주지는 않음
  - `[PL3]`은 NLR profile의 측정·확대 근거를 제공하지만 현재 파일 각각이 물리적으로 몇 개 서버 또는 몇 개 workload unit을 대표하는지는 추가 확인이 필요함
  - 따라서 문헌에서 직접 식별되지 않는 $\rho$와 $N_{\mathrm{eff}}$를 최종 물리 파라미터로 주장하지 않기로 함

---

## 10. R2/R3 병렬 workload-group aggregation v2

- Workload group별 M3 정상상태 전력과 R2 또는 R3 residual을 생성한 뒤 동일 용량가중치로 직접 합산

$$
P_{\mathrm{IT}}(t)
=s\sum_{g=1}^{G}w_g
\left[
P_{\mathrm{M3}}(r_g,d_g)
+e_{m,g}(t\mid r_g,d_g)
\right]
,
\qquad
\sum_{g=1}^{G}w_g=1,
\quad
m\in\{\mathrm{R2},\mathrm{R3}\}
$$

- $s$: 전체 평균 IT 전력을 20 MW로 맞추는 scale factor
- $G$: equivalent workload-group 수. 현재는 실제 서버/GPU 수가 아닌 민감도 변수
- $w_g=1/G$: 현 단계에서 적용한 동일 용량가중치
- $r_g$, $d_g$: group별 request rate와 prompt dataset
- $e_{m,g}$: R2 OU 또는 R3 Jump-OU residual

### 10.1. 시나리오 구성

| 구분 | 적용값 |
|---|---|
| 평균 IT 전력 | 20 MW |
| 분석시간·해상도 | 180 s, 0.1 s |
| Request rate | 10, 50, 90 request/s |
| Prompt dataset | InstructCoder, mlperf-inference-llama2-data |
| Residual model | R2 OU, R3 Jump-OU |
| Equivalent group 수 | 1, 10, 50 |
| 동기성 | Independent, partial 25%, fully synchronized |
| Monte Carlo | 조건별 10회, 총 1,080 scenarios |

- Independent: 각 group이 서로 다른 residual realization을 사용
- Partial 25%: 전체 group 중 약 25%가 동일 residual realization을 공유
- Fully synchronized: 모든 group이 같은 residual realization을 공유하는 stress case
- Partial 25%는 $G=50$일 때 반올림으로 12개 group, 즉 실제 동기비율 24%로 구현
- 이 동기성은 현재 NLR에서 식별한 파라미터가 아니라 **명시적인 sensitivity/stress scenario**임

### 10.2. 90 request/s, 50 group 결과

| Model·dataset | 동기성 | 표준편차 (MW) | P01–P99 import (MW) | P99 absolute ramp (MW/s) |
|---|---|---:|---:|---:|
| R2·InstructCoder | Independent | 0.0881 | 19.80–20.20 | 0.678 |
| R2·InstructCoder | Partial 25% | 0.1717 | 19.58–20.39 | 1.286 |
| R2·InstructCoder | Fully synchronized | 0.6792 | 18.45–21.62 | 4.864 |
| R3·InstructCoder | Independent | 0.0965 | 19.78–20.23 | 0.727 |
| R3·InstructCoder | Partial 25% | 0.1832 | 19.57–20.43 | 1.987 |
| R3·InstructCoder | Fully synchronized | 0.6934 | 18.29–21.67 | 8.109 |
| R2·Llama2 | Independent | 0.1085 | 19.74–20.24 | 0.691 |
| R2·Llama2 | Partial 25% | 0.2197 | 19.47–20.49 | 1.293 |
| R2·Llama2 | Fully synchronized | 0.7934 | 18.23–21.73 | 4.846 |
| R3·Llama2 | Independent | 0.1136 | 19.74–20.26 | 0.747 |
| R3·Llama2 | Partial 25% | 0.2260 | 19.47–20.50 | 2.165 |
| R3·Llama2 | Fully synchronized | 0.7730 | 18.02–21.82 | 8.792 |

### 10.3. 결과 해석

- Independent 조건에서는 group 수 증가에 따라 개별 residual이 평균화되어 변동성과 ramp가 감소
- Fully synchronized 조건에서는 group 수가 증가해도 공통변동이 상쇄되지 않음
- R2와 R3의 P01–P99 전력범위 차이는 상대적으로 작지만, R3는 partial/full synchronization에서 P99 ramp를 더 크게 추정
- 즉, 평균 전력 또는 정적 import envelope만 보면 R2·R3 차이를 작게 볼 수 있으나 ramp 제약을 포함한 DOE에서는 모델 선택이 달라질 가능성이 있음
- 이 결과가 R2/R3 최종 선정을 DOE violation과 decision regret까지 미루는 직접적 이유임

![R2/R3 20 MW 대표 profile](figures/representative_r2_r3_20mw_profiles.png)

> Figure. 90 request/s, 50 equivalent workload-group에서 R2와 R3 및 동기성 시나리오에 따른 20 MW IT profile 비교.

![R2/R3 P01-P99 import envelope](figures/r2_r3_uncertainty_envelopes.png)

> Figure. DSO 전달정보 후보인 P01–P99 IT import envelope. 동기성이 높아질수록 운전범위가 넓어진다.

![R2/R3 ramp 비교](figures/r2_r3_ramp_by_group_count.png)

> Figure. Equivalent group 수와 동기성에 따른 P99 absolute ramp. R3가 동기 시나리오에서 더 큰 ramp tail을 보존한다.

### 10.4. DSO 전달정보 후보

- 예상 평균 IT import
- P01–P99 IT import envelope
- P95/P99 absolute ramp
- 상향 유연성 필요량: $P_{\max}-P_{\mathrm{mean}}$
- 하향 유연성 필요량: $P_{\mathrm{mean}}-P_{\min}$
- 1 s 및 10 s ACF

> 현재 값은 **IT 부하 정보**이다. 실제 PCC import 정보를 구성하려면 cooling, UPS loss, auxiliary load 및 ESS–TES action을 추가해야 한다.

- 근거 문헌
  - Paper list [3]의 bottom-up facility simulation은 workload profile을 구성요소 수준에서 합산한다는 기본 방향을 제공함 `[PL3]`
  - `[EXT5]`는 여러 서버의 실제 소비전력을 단순 nameplate 합이 아닌 측정 기반 aggregate profile로 평가해야 한다는 근거를 제공함
  - `[EXT3]`은 공통성분과 개별성분을 구분해 상관구조를 유지해야 한다는 방법론적 근거를 제공함
  - `[PL17]`은 비동기 replica의 random time shift와 중첩을 사용한 AIDC-specific 예를 제공하지만 preprint이므로 보조근거로만 사용함
  - 직접 가중합 식, equal-weight group, partial 25% 시나리오 및 20 MW 결과는 **본 연구의 구현·민감도 분석**이며 문헌 원식을 그대로 사용한 것이 아님

---

## 11. 현재까지의 완료·검토 상태

- 아래 완료 여부는 코드·notebook·출력파일을 기준으로 정리한 **본 연구 진행상태**임
- 각 분석단계의 외부 근거는 앞 절의 `[PL#]`, `[EXT#]` 문헌을 따름

### 완료

- NLR `online inference rate` condition-level 정상상태 dataset 구성
- M1·M2·M3 정상상태식 비교
- Prompt-dataset-conditioned M3 선정 및 parameter 추정
- M3 residual의 variability·distribution·ACF·ramp·PSD·innovation-tail 분석
- Train-only OU/innovation/GMM 구조진단 수행
- R0–R4 residual 후보 비교
- R2를 provisional statistical baseline으로 선정하고 R3 tail-aware 후보를 병렬 유지
- 재사용 가능한 M3+R2/R3 equivalent workload-group generator 구축
- 출처가 불명확한 $\sqrt{\rho}$ common-factor 집합화식 폐기
- Direct group summation 기반 20 MW IT aggregation v2 1,080개 scenario 수행
- DSO 전달정보 후보 table 생성

### 신규 구현코드·산출물

- `src/nlr_data/residual_diagnostics.py`
- `src/nlr_data/residual_model_selection.py`
- `src/nlr_data/workload_aggregation_v2.py`
- `notebooks/08a_nlr_residual_diagnostics.ipynb`
- `notebooks/08b_nlr_residual_model_selection.ipynb`
- `notebooks/09_nlr_workload_aggregation_v2.ipynb`
- `outputs/residual_diagnostics/`
- `outputs/residual_model_selection_v2/`
- `outputs/workload_aggregation_v2/`

### 추가 검토 필요

- M3 수식의 peer-reviewed 직접 근거 또는 문헌 원식 기반 대체모델 선정
- NLR profile이 대표하는 물리적 집합단위의 명확화
- Workload group 수와 용량가중치 $w_g$의 물리적 정의
- 동시성·cross-correlation 구조의 실제 데이터 기반 추정 가능성 검토
- R2와 R3가 DSO의 DOE 및 운영결정에 주는 차이 검증
- 최종 facility IT profile 확정 후 cooling·UPS loss·보조부하 결합

---

## 12. 다음 진행 예정 사항

- 1단계: 20 MW IT profile에 cooling·UPS loss·보조부하를 결합하여 PCC import model 구성
- 2단계: Modified IEEE 33-bus의 최소 OPF/DOE 산정환경 구축
- 3단계: 동일 forecast information에서 R2와 R3가 산정한 import/ramp uncertainty를 각각 DSO에 전달
- 4단계: R2/R3 기반 DOE의 voltage·thermal·ramp constraint violation 비교
- 5단계: 완전정보 기준 DOE와 비교하여 decision regret 계산
- 6단계: R2 통계 baseline과 R3 tail-aware model 중 계통운영 목적에 적합한 family 결정
- 7단계: 선정된 stochastic gray-box 정보를 일반 예측학습과 DFL에 각각 연결
- 8단계: 일반학습 대비 DFL의 DOE violation·decision regret 개선효과 평가
- 9단계: ESS–TES local dispatch 및 OPAL-RT DOE 지령추종 검증으로 확장

### 바로 다음 구현단계

$$
P_{\mathrm{PCC}}(t)
=P_{\mathrm{IT}}(t)
+P_{\mathrm{cool}}(t)
+P_{\mathrm{aux}}(t)
+P_{\mathrm{UPS,loss}}(t)
-P_{\mathrm{BESS}}(t)
$$

- 현재의 $P_{\mathrm{IT}}$가 전체 AIDC PCC 전력으로 잘못 해석되지 않도록 구성요소별 model boundary를 먼저 정의
- 그다음 최소 배전계통에서 정적 import DOE와 ramp-aware DOE를 분리하여 구현
- 이 단계 전에는 R2 또는 R3를 최종 family로 고정하지 않음

- 향후 확장 관련 문헌
  - Paper list [11]: PSCAD 기반 데이터센터 EMT model과 workload·UPS/PFC·보호 동특성 확장 근거 `[PL11]`
  - Paper list [15]: cooling/thermal gray-box model과 data-driven parameter correction 근거 `[PL15]`
  - Paper list [16]: IT/non-cooling과 cooling을 분리한 data-physics hybrid total-load 구성 근거 `[PL16]`
  - 위 문헌은 향후 확장 방향의 근거이며 현재 M3+R2/R3 IT generator가 PCC 전체모델까지 완료되었다는 의미는 아님

---

## 참고문헌

### Paper list 문헌

- **[PL2]** S. Lu, C. Xiao, and Y. Weng, “Dynamic Load Model for Data Centers with Pattern-Consistent Calibration,” arXiv:2602.07859, 2026. [원문](https://arxiv.org/abs/2602.07859) — **preprint**
- **[PL3]** R. Vercellino et al., “Measurement of Generative AI Workload Power Profiles for Whole-Facility Data Center Infrastructure Planning,” arXiv:2604.07345, 2026. [원문](https://arxiv.org/abs/2604.07345) — **NLR 데이터 출처, preprint**
- **[PL9]** H. Cheung, S. Wang, C. Zhuang, and J. Gu, “A simplified power consumption model of information technology (IT) equipment in data centers for energy system real-time dynamic simulation,” *Applied Energy*, vol. 222, pp. 329–342, 2018. [DOI](https://doi.org/10.1016/j.apenergy.2018.03.138) — **peer-reviewed**
- **[PL11]** B. A. Ross and J. Follum, “Electromagnetic Transient Modeling of Large Data Centers for Grid-Level Studies,” PNNL-38817, Pacific Northwest National Laboratory, 2026. [공식 페이지](https://www.pnnl.gov/publications/electromagnetic-transient-modeling-large-data-centers-grid-level-studies) — **technical report**
- **[PL15]** D. Chen, C.-K. Chui, and P. S. Lee, “Adaptive physically consistent neural networks for data center thermal dynamics modeling,” *Applied Energy*, vol. 377, 124637, 2025. [DOI](https://doi.org/10.1016/j.apenergy.2024.124637) — **peer-reviewed**
- **[PL16]** Z. Zhao et al., “Short-term data center load forecasting method based on data-physics hybrid driven framework,” *Energy Reports*, vol. 15, 108953, 2026. [DOI](https://doi.org/10.1016/j.egyr.2025.108953) — **peer-reviewed**
- **[PL17]** Z. Liang, J.-W. Chung, M. Chowdhury, J. Chen, and V. Dvorkin, “GPU-to-Grid: Voltage Regulation via GPU Utilization Control,” arXiv:2602.05116, 2026. [원문](https://arxiv.org/abs/2602.05116) — **preprint**
- **[PL37]** S. Majumder, M. Yu, and L. Xie, “Workload composition smooths aggregate power demand while sustaining short-horizon ramps in AI data centers,” arXiv:2604.10769, 2026. [원문](https://arxiv.org/abs/2604.10769) — **preprint**
- **[PL38]** A. Jiménez-Ruiz and F. Milano, “Data Center Model for Transient Stability Analysis of Power Systems,” *IET Generation, Transmission & Distribution*, 2026. [DOI](https://doi.org/10.1049/gtd2.70381) — **peer-reviewed; Paper list [1] preprint의 정식 저널버전, OU residual 및 workload jump/pulse 모델 근거**

### 외부 문헌

- **[EXT1]** M. Dayarathna, Y. Wen, and R. Fan, “Data Center Energy Consumption Modeling: A Survey,” *IEEE Communications Surveys & Tutorials*, vol. 18, no. 1, pp. 732–794, 2016. [DOI](https://doi.org/10.1109/COMST.2015.2481183) — **peer-reviewed**
- **[EXT2]** D. R. Roberts et al., “Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure,” *Ecography*, vol. 40, no. 8, pp. 913–929, 2017. [DOI](https://doi.org/10.1111/ecog.02881) — **peer-reviewed**
- **[EXT3]** D. Lee and R. Baldick, “Load and Wind Power Scenario Generation Through the Generalized Dynamic Factor Model,” *IEEE Transactions on Power Systems*, vol. 32, no. 1, pp. 400–410, 2017. [DOI](https://doi.org/10.1109/TPWRS.2016.2562718) — **peer-reviewed**
- **[EXT4]** R. Sevlian and R. Rajagopal, “A scaling law for short term load forecasting on varying levels of aggregation,” *International Journal of Electrical Power & Energy Systems*, vol. 98, pp. 350–361, 2018. [DOI](https://doi.org/10.1016/j.ijepes.2017.10.032) — **peer-reviewed**
- **[EXT5]** X. Fan, W.-D. Weber, and L. A. Barroso, “Power Provisioning for a Warehouse-Sized Computer,” in *Proceedings of the 34th Annual International Symposium on Computer Architecture*, pp. 13–23, 2007. [DOI](https://doi.org/10.1145/1273440.1250665) — **peer-reviewed conference paper**
