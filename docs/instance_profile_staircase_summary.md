# IT 부하 프로파일 재구성 및 계단형 파형 원인 진단

> 진행 범위: instance 기반 시간가변 요청 분배 → M3+R3 시설 IT 합산 → 계단형 파형 발견 → 집계 간격별 진단 → NLR 원시 로그 점검
>
> 현재 결론: 시간가변 수요와 전력을 연결하는 구조는 구현했으나, 빠른 전력 전이응답은 아직 검증하지 못함. 근거 없는 smoothing이나 시간상수는 추가하지 않음.

## 1. 기존 IT 부하 합산 구조의 한계 확인

- 기존 방식: 개별 단위마다 prompt dataset과 request rate를 정한 뒤, 180초 동안 같은 조건으로 M3+R3 프로파일을 생성하여 합산
  - 서로 다른 단위의 request rate는 다를 수 있으나, 각 단위의 rate는 시간에 따라 변하지 않음
  - 독립적인 residual 변동이 합산 과정에서 상쇄되어 시설 전력의 변동폭이 작게 나타남
- 보완이 필요한 부분: **서비스 전체 요청량이 시간에 따라 달라지면서 평균전력이 이동하는 과정**이 포함되지 않음
  - 단순히 residual의 크기를 키우거나 서버 간 임의의 동기화를 넣는 방식은 사용하지 않음
  - 기존 결과는 ‘고정 수요·독립 변동’의 기준선으로 보존
- 작은 변동폭 자체가 비현실적이라는 결론은 아님
  - 기존 모델이 표현하는 상황과 시간가변 수요를 표현하는 상황을 구분할 필요가 있음

**→ 개별 전력 변동을 인위적으로 키우기보다, 전력의 원인이 되는 요청량의 시간 변화를 먼저 반영함.**

## 2. Instance 기반 요청 분배 및 전력 합산 구조 구성

- Instance: 요청을 처리하기 위해 실행 중인 **LLM serving 실행 단위**
  - 개별 요청(request) 자체를 뜻하지 않음
  - 한 instance가 여러 GPU를 사용할 수 있으므로, 일반적으로 instance와 물리 서버가 반드시 일대일인 것은 아님
- 현재 구현: NLR 실험을 참고하여 **node/instance 등가 단위 수를 고정**하고 서비스별 요청량을 균등 분배*[3]
  - instance 수를 시간에 따라 늘리거나 줄이는 autoscaling은 이번 구현에 포함하지 않음
  - 대기열·요청 완료량·응답시간 모델도 아직 포함하지 않음

**시간가변 서비스 요청량 → 고정 instance들에 분배 → instance별 M3+R3 전력 생성 → 시설 IT 전력 합산**

$$
r_{d,i}(t)=\frac{\Lambda_d^{\mathrm{admitted}}(t)}{N_d},
\qquad
P_{\mathrm{IT}}(t)=\sum_d\sum_{i=1}^{N_d}
\left[P_{\mathrm{M3}}\!\left(r_{d,i}(t),d\right)+e_{d,i}(t)\right].
$$

- $d$: prompt dataset에 대응시킨 서비스 종류
- $N_d$: 해당 서비스에 배정한 고정 instance 등가 단위 수
- $\Lambda_d^{\mathrm{admitted}}(t)$: 서비스에 제출하는 전체 요청률; 실제 처리 완료율과는 다름
- $r_{d,i}(t)$: 개별 instance의 입력 요청률
- $P_{\mathrm{M3}}$: 해당 요청률의 평균 운전점, $e_{d,i}$: R3 residual
- 요청률이 바뀌어도 residual 상태를 이어서 생성하며, 구간마다 재시작하거나 평균을 다시 0으로 맞추지 않음

\* **차용 범위:** [3]의 요청 분배·전력 배정·합산 흐름을 참고하되, 원래의 실측 전력 sample 배정을 기존 M3+R3로 대체한 **본 연구의 재구성**임. 위 전체 식을 [3]에서 그대로 가져온 것은 아님. M3/R3의 기존 방법론 근거와 수정 범위는 하단 각주에 별도 명시.

| 설정 항목 | 이번 실험의 구성 |
|---|---|
| 합산 단위 수 | 6,487개 등가 node/instance |
| 코드 생성 서비스 | InstructCoder 1,946개, 약 30% |
| 대화 서비스 | Llama2-data 4,541개, 약 70% |
| 생성 시간 / 출력 간격 | 180초 / 0.1초 |
| 최초 요청량 집계 간격 | 10초 |
| 기준 IT 전력 | $20/1.15 \simeq 17.39$ MW |

- 30/70은 **단위 수 구성비**이며 요청량 또는 전력 구성비가 아님
- 6,487개는 시나리오의 등가 단위 수이며 실측으로 확인한 시설 서버 대수가 아님
- PUE 1.15는 기준 IT 규모를 정하는 산술에만 사용; cooling 및 PCC 프로파일을 생성한 것은 아님
- 출력 경계는 NLR 후처리가 보고한 CPU/GPU 합산 전력에 기반함
  - full-server AC/PCC 계측값으로 검증된 것은 아니며, CPU package/core 채널 중첩 여부도 추가 확인 필요

## 3. 실제 요청 기록을 이용한 180초 IT 프로파일 생성

- Azure LLM inference 공개 데이터에서 coding·conversation 서비스의 같은 180초 구간을 추출*[44]
  - 원자료 요청 수: coding 2,596건 / conversation 4,557건
  - 원자료의 시간축을 유지하고, 요청률의 크기만 시설 시나리오에 맞게 조정
  - 시간·일 단위 패턴을 180초로 압축하지 않음
  - 해당 자료는 **요청 시각과 token 기록**이며 실제 시설 전력 데이터는 아님
- 시간가변 요청량의 영향을 분리하기 위해 B·C 조건을 비교
  - **B:** 서비스별 평균 요청량을 180초 동안 일정하게 유지
  - **C:** B와 같은 평균·총 입력 요청량을 유지하되, 요청량의 시간적 분포를 원자료에 따라 변경
  - node 수·분배 방식·M3/R3 계수는 동일하게 유지
  - 각 조건에서 같은 traffic에 residual 5회 생성; 서로 다른 실측 traffic 5개가 아님

| 비교 조건 | 평균 IT 전력 | 시간축 표준편차 |
|---|---:|---:|
| B: 고정 요청량 | 17.3917 MW | 2.50 kW |
| C: 시간가변 요청량 | 17.3536 MW | 171.00 kW |

*표의 수치는 5개 생성 시나리오에서 계산한 지표의 평균. 기존 독립 합산안은 rate 배분·residual 처리도 달라 직접적인 대조군으로 사용하지 않음.*

![그림 1. 요청량 변화와 IT 평균 운전점 및 residual의 연결](D:/DC_modeling/aidc-dfl-doe-research/outputs/time_varying_it_v1/traffic_to_power.png)

**그림 1 설명**

- 상단: 10초 단위로 집계·분배한 instance당 요청률
- 중단: 고정 요청량 B와 시간가변 요청량 C의 시설 전력 예시
- 하단: 평균 운전점에서 벗어나는 합산 residual
- C의 변동은 대부분 **M3 평균 운전점 이동**에서 발생
  - M3 합산 표준편차 약 170.95 kW / residual 합산 표준편차 약 3.79 kW
  - 독립 residual의 상쇄를 없앤 것이 아니라, 공통적인 서비스 수요 변화를 입력으로 추가한 결과
- B와 C의 평균전력 차이 약 38.10 kW는 비선형 M3 및 유한 residual 실현의 결과
  - 출력 평균을 17.39 MW로 강제 보정하지 않았음
  - 완료 요청량·응답시간을 검증하지 않았으므로 에너지 절감 성과로 해석하지 않음

**→ 요청량이 변하면 시설 평균전력도 변하는 구조는 구현했으나, 평균전력이 계단형으로 바뀌는 문제가 드러남.**

## 4. 계단형 파형의 원인 확인

- **요청량을 10초 동안 일정한 값으로 유지**한 뒤 다음 구간에서 변경
- 정적 모델 M3가 변경된 request rate에 해당하는 평균전력으로 **즉시 이동**
- 출력은 0.1초 간격이므로, 구간 경계의 전력 차이가 0.1초 사이의 급격한 변화로 기록됨
  - `0.1초 출력 간격`과 `10초 요청량 집계 간격`은 서로 다른 설정
  - 0.1초 간격으로 출력했다고 실제 0.1초 시설 전이응답을 재현한 것은 아님
- R3는 운전점 주변의 확률적 변동을 표현하며, 요청률 변경에 대한 **평균전력의 응답 지연을 자동으로 표현하지 않음**
- 이번 구현의 node/instance 수는 고정
  - 과거의 ‘active node 수가 동시에 바뀌는 문제’와 달리, 이번 계단은 **입력 집계와 M3 즉시응답 연결**에서 발생

**→ 문제는 instance라는 개념 자체가 아니라, 고정 운전점에서 보정한 전력식을 시간가변 입력에 즉시 적용한 방식에 있음.**

## 5. 동일 요청자료의 집계 간격·경계 위치 진단

- 가설: “요청량 집계 간격을 줄이면 계단 문제가 완화되는가?”
- 같은 180초 요청 기록에 대해 집계 간격을 **10초 → 5초 → 1초 → 0.1초**로 변경
  - 단위 수·수요 배율·M3/R3 계수·출력 간격·총 요청량은 유지
  - 집계 시작 경계도 반 구간씩 이동하여 입력 표현에 대한 민감도 확인
- 10·5·1초 조건은 전력 생성 및 비교 수행
  - 0.1초 조건은 0 request/s 또는 기존 지원범위 밖의 입력이 발생하여 전력 생성에서 제외
  - 범위 밖 입력을 임의로 잘라내거나 idle 전력으로 대체하지 않음

| 요청 집계 간격 | M3 합산 평균 | M3 합산 표준편차 | 최대 절대 0.1초 ramp |
|---|---:|---:|---:|
| 10초 | 17.3534 MW | 170.95 kW | 2.77 MW/s |
| 5초 | 17.3281 MW | 226.17 kW | 5.99 MW/s |
| 1초 | 17.1414 MW | 564.02 kW | 22.98 MW/s |

*표는 집계 경계를 이동하지 않은 조건의 **M3 합산만** 비교한 값. 최대 ramp에는 구간 경계의 즉시 이동이 포함되므로 실제 시설 ramp 추정치가 아님.*

![그림 2. 동일 요청 기록의 집계 간격 및 경계 위치에 따른 M3 전력 차이](D:/DC_modeling/aidc-dfl-doe-research/outputs/traffic_bin_diagnostic_v1/bin_phase_power_comparison.png)

**그림 2 설명**

- 위에서부터 10초·5초·1초 집계 결과
- 실선과 점선은 같은 요청 기록에서 집계 경계만 반 구간 이동한 결과
- 집계 간격을 줄여도 전력이 부드럽게 수렴하지 않고, 더 잦고 큰 변화가 나타남
  - 짧은 구간의 요청 개수 요동이 확대되어 정적 M3에 즉시 반영되기 때문
- 집계 경계 위치만 바꾸어도 파형이 달라짐
  - 현재 빠른 ramp는 실제 요청 기록만으로 고유하게 결정되는 값이 아님

**→ 집계 간격을 줄이는 것만으로는 해결되지 않음. 요청의 도착과 실제 처리상태·전력의 전이를 연결할 근거가 필요함.**

## 6. NLR 원시 로그에서 전이응답 식별 가능성 확인

- 목적: 새로운 시간상수를 임의로 넣기 전에, 기존 NLR 자료에서 요청률 변화에 대한 전력 응답을 추정할 수 있는지 확인
- 점검 범위: 로컬 online rate·online finite·offline inference의 **17개 파일, 약 0.85 GB**
  - 기존 180초 추출 구간뿐 아니라 원시 로그의 실행 시작·종료 주변까지 확인

| 확인 항목 | 확인 결과 | 의미 |
|---|---|---|
| 원시 CPU/GPU 전력 | 약 0.1초 간격 | 전력 변화 자체는 관찰 가능 |
| Running/Waiting 등 서버 상태 | online 자료에서 대체로 10초 간격 | 빠른 처리상태 전이 분석에는 제한 |
| Online rate 실험 설정 | 200회 실행, 실행별 고정 request rate | 고정 운전점 분석 자료로 활용 가능 |
| 요청별 정밀 도착 시각 | 점검한 HTTP access 행에서 미확인 | 요청 변화와 전력 변화의 정밀 대응이 어려움 |
| 가동 중 명확한 요청률 step | 점검 자료에서 미확인 | 일반적인 운전점 전이시간을 확정하기 어려움 |

![그림 3. NLR online rate 실행 경계 주변의 원시 GPU 전력](D:/DC_modeling/aidc-dfl-doe-research/outputs/nlr_transition_audit_v1/inference_online_rate_llama3_70b_boundary_diagnostics.png)

**그림 3 설명**

- 파란색: GPU 채널 합을 시각 점검용으로 1초 평균한 전력
- 주황색: 상대적으로 성긴 Running request 기록
- 검은 점선: benchmark 실행 경계이며 정확한 요청률 step 시각은 아님
- 시작·종료 주변의 전력 변화는 확인되지만, 설정·단일 요청 시험·처리 종료 과정의 영향을 분리하기 어려움
  - 해당 시간을 그대로 “request rate 변경에 대한 응답 지연”으로 사용하지 않음
  - full-server/PCC 전력이나 시설 합산 결과를 나타내는 그림이 아님

**→ 고정 요청률 자료와 실행 시작·종료 분석 후보는 있으나, 현재 점검한 자료만으로 일반적인 가동 중 전이시간을 검증하기에는 근거가 부족함.**

## 7. 현재까지의 결론 및 후속 방향

- **완료한 작업**
  - 고정 instance 기반 시간가변 요청 분배 및 M3+R3 시설 합산
  - 계단형 파형의 원인 분리
  - 동일 요청자료의 집계 간격·경계 민감도 분석
  - NLR 원시 로그의 전이응답 식별 가능성 점검
- **아직 완료하지 않은 작업**
  - 새로운 전이모델의 선정 및 시간상수·지연 추정
  - 계단을 제거한 프로파일의 물리적 검증
  - 실제 facility IT/PCC 전력과의 고속 ramp 비교
- **현재의 처리 원칙**
  - 기존 M3+R3는 고정 운전점 및 준정상 비교 기준으로 보존
  - 파형을 보기 좋게 만들기 위한 임의 smoothing은 적용하지 않음
  - R3의 상관시간을 평균전력 전이시간으로 대체하지 않음
- **다음에 필요한 근거**
  - 같은 가동 중 instance에서 요청률을 바꾸면서 요청 송신·처리 시작·완료 시각과 전력을 함께 기록한 자료
  - 자료 확보 후 최소 전이모델의 필요성과 파라미터 식별 가능성을 검토
  - 대기열·동시 처리 요청 수가 필요하면 [43]의 요청 → 처리상태 → 전력 접근을 후속 대안으로 검토하되, 현재 구현된 구조로 표현하지 않음*[43]

**보고용 한 줄 요약**

> 시간가변 요청량을 instance별 M3+R3에 연결해 시설 IT 부하를 재구성했으며, 계단형 파형의 원인이 요청 집계와 정적 운전점의 즉시 변경에 있음을 확인하였다. 집계 간격 축소만으로는 문제가 해결되지 않았고, NLR 원시 로그에서도 전이시간을 식별할 근거가 부족하여 추가 전이모델 선정은 보류하였다.

## Reference 각주 및 차용 범위

- \* **[3]** Vercellino et al., *Measurement of Generative AI Workload Power Profiles for Whole-Facility Data Center Infrastructure Planning*, arXiv, 2026. [원문](https://arxiv.org/html/2604.07345v1)
  - 요청량 → instance 분배 → 전력 배정 → 시설 합산 흐름과 NLR 측정자료의 근거
  - 시설 inference 사례는 1분 시간간격이며, 본 연구의 0.1초 시설 ramp 검증 근거가 아님
  - 고정 단위 수·M3+R3 대체·시간가변 residual 연결은 본 연구의 구현 또는 가정
- \* **기존 M3/R3 방법론:** [37] *Workload composition smooths aggregate power demand while sustaining short-horizon ramps in AI data centers*의 평균 성분+residual 접근, [2] *Dynamic Load Model for Data Centers with Pattern-Consistent Calibration*의 Jump-OU 골격을 참고한 기존 구현을 재사용
  - R3의 이산 signed-mixture innovation은 본 연구의 근사이며 원 논문의 식과 동일하지 않음
  - 이번 작업은 M3/R3 모델 재선정이 아니라 입력·분배·시간 연결의 확장임. 기존 연결 근거는 [고정 생성기 설명](D:/DC_modeling/aidc-dfl-doe-research/src/nlr_data/final_it_profile_generator.py:6)과 [benchmark protocol](D:/DC_modeling/aidc-dfl-doe-research/outputs/r3_high_rate_benchmark/protocol.json)에 기록되어 있음
- \* **[44]** *DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency*, IEEE HPCA, 2025. [DOI](https://doi.org/10.1109/HPCA61900.2025.00102), [Azure 공식 데이터 설명](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md)
  - 실제 요청 기록의 출처. 해당 논문의 운영 최적화나 실제 시설 전력까지 재현한 것은 아님
- \* **[43]** Wilkins et al., *From Servers to Sites: Compositional Power Trace Generation of LLM Inference for Infrastructure Planning*, arXiv, 2026. [원문](https://arxiv.org/html/2603.18383v1)
  - 요청·처리상태 기반 전력 구성의 보조 근거 및 향후 대안
  - 현재 참고한 판본은 preprint이며, 논문의 queue·상태 분류 모델을 이번에 구현하지 않음
- **본 연구에서 구성한 진단:** 동일 요청자료의 집계 간격·경계 민감도 비교와 전이자료 적합성 판단 절차. 특정 논문의 검증 프로토콜을 그대로 재현한 것은 아님

## 상세 결과 위치

- [Instance 기반 시간가변 프로파일 실행 결과](D:/DC_modeling/aidc-dfl-doe-research/outputs/time_varying_it_v1/report_ko.md)
- [계단형 파형 진단 상세 결과](D:/DC_modeling/aidc-dfl-doe-research/outputs/it_transition_diagnosis_v1/report_ko.md)
- [측정 경계 및 기존 모델 감사](D:/DC_modeling/aidc-dfl-doe-research/docs/time_varying_it_model_audit.md)

*문서의 그림은 기존 결과 PNG를 연결함. 외부 Notion 페이지로 옮길 때 로컬 경로만으로는 그림이 업로드되지 않으므로 해당 PNG를 함께 첨부해야 함.*
