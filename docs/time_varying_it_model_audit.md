# 시간가변 IT 부하 모델 확장을 위한 측정 경계 및 기준선 감사

- 단계: 기존 M3+R3 생성기를 시간가변 request rate와 연결하기 전의 근거 확인
- 범위: 기존 코드, NLR 원자료의 설정·로그, 저장된 parameter 및 결과 파일을 읽어 확인
- 이 문서는 기존 실험을 다시 실행하거나 모델 성능을 새롭게 검증한 보고서가 아님
- 핵심 원칙: 현재 독립 합산 결과를 기준선으로 보존하고, 확인된 측정 경계 안에서 운영 입력만 확장함

## 1. 먼저 확정할 해석

현재 M3+R3는 NLR의 특정 inference serving 실험 조건에서 관측한 전력과 변동을 재현하는 생성기임. 입력의 `request_rate`는 **benchmark에 설정한 요청 도착률**이며, 실제 처리 완료율을 뜻하지 않음. 출력은 **NLR 후처리가 보고한 CPU/GPU 합산 전력**이며, 완전한 서버 입력전력이나 시설 PCC 계측값이 아님.

NLR [3]은 한 compute node에서 inference model instance를 실행한 사례를 제공하지만, 이 사실만으로 기존 등가 합산 결과의 6,487 unit을 검증된 물리 서버 수라고 바꾸어 부르면 안 됨. 원자료와 특정 하드웨어의 대응, 포함 전력 채널, 시설 집계 가정을 함께 명시해야 함.*

## 2. NLR 입력·측정 경계 확인 결과

| 확인 항목 | 확인된 내용 | 파일 근거 |
|---|---|---|
| Request rate의 의미 | `vllm bench serve`의 입력 인자. `request_rate=1.0`, `num_prompts=300`, `burstiness=100000.0` 등으로 설정됨. 완료 throughput으로 보정한 입력이 아님 | [vllm_bench.log](D:/DC_modeling/NLR_dataset/00_raw_datasets/inference_online_rate_llama3_70b/vllm_bench.log:4), [config.json](D:/DC_modeling/NLR_dataset/00_raw_datasets/inference_online_rate_llama3_70b/config.json:8) |
| 실제 모델 구성 | 로그에는 `Llama-3.1-70B-Instruct`, tensor parallel 4, pipeline parallel 1, data parallel 1이 기록됨. 논문의 Llama-3 계열 표기와 구체적인 실행 artifact를 구분함 | [vllm_server.log](D:/DC_modeling/NLR_dataset/00_raw_datasets/inference_online_rate_llama3_70b/vllm_server.log:10) |
| GPU 채널 | 동일 host의 GPU 0~3 전력 채널 4개 | [NVML 로그](D:/DC_modeling/NLR_dataset/00_raw_datasets/inference_online_rate_llama3_70b/nvml_wattameter_12146821-x3104c0s13b0n0.log:3) |
| CPU 채널 | 같은 host의 CPU 0·1에 대해 package 및 core 이름의 전력 채널이 있음. 전력 열 4개를 CPU 4개로 해석하지 않음 | [RAPL 로그](D:/DC_modeling/NLR_dataset/00_raw_datasets/inference_online_rate_llama3_70b/rapl_wattameter_12146821-x3104c0s13b0n0.log:3) |
| 원자료 합산 | NLR 후처리는 GPU 4개와 CPU package/core 4개 열을 합해 `power[W]`를 생성함 | [postprocess.py](D:/DC_modeling/NLR_dataset/01_aggregated_datasets/inference_online_rate_llama3_70b/postprocess.py:164), [합산 코드](D:/DC_modeling/NLR_dataset/01_aggregated_datasets/inference_online_rate_llama3_70b/postprocess.py:179) |
| 180초 구간 | 실행 시작 후 1분을 건너뛴 뒤 최대 3분을 추출함. `num_prompts=request_rate×300`인 설정과 분석 구간 180초는 서로 다른 항목임 | [구간 추출 코드](D:/DC_modeling/NLR_dataset/01_aggregated_datasets/inference_online_rate_llama3_70b/postprocess.py:267), [metadata.csv](D:/DC_modeling/NLR_dataset/01_aggregated_datasets/inference_online_rate_llama3_70b/metadata.csv:2) |
| 시간해상도 | raw 로그의 간격은 대략 0.1초이며 NLR 후처리에 0.001초 재표본화가 포함됨. 따라서 aggregated 파일의 1ms 간격을 1ms 실측으로 해석하지 않음 | [재표본화 코드](D:/DC_modeling/NLR_dataset/01_aggregated_datasets/inference_online_rate_llama3_70b/postprocess.py:160), [현재 생성 간격](D:/DC_modeling/aidc-dfl-doe-research/src/nlr_data/final_it_profile_generator.py:47) |

### CPU package/core 합산에 대한 유보사항

- 위 합산은 **NLR 원본 후처리를 그대로 따랐다는 확인**임.
- CPU package와 core 채널이 물리적으로 중첩되는 domain인지, 이 계측 환경에서 상호 배타적인 값인지까지 확인한 것은 아님.
- 따라서 중복 집계 가능성을 배제할 수 없지만, 현재 증거만으로 **중복 집계가 발생했다고 확정하지도 않음**.
- 향후 WattAMeter/RAPL의 해당 장비·backend별 채널 정의 또는 연구진 확인이 필요함. 확인 전에는 임의로 열을 제거하거나 기존 calibration을 다시 수행하지 않음.
- 현재 출력은 `upstream-reported CPU/GPU aggregate power`로 기술하는 것이 안전함. RAM·NIC·팬·전원공급장치 손실 등을 포괄하는 독립적인 full-server AC meter 측정으로 검증되지 않았음.

## 3. Offered·admitted·completed request를 구분

| 용어 | 의미 | 새 모델에서의 취급 |
|---|---|---|
| Offered arrival rate | 시설 또는 서비스에 들어오려고 하는 요청의 도착률 | 외부 traffic 입력 |
| Admitted arrival rate | 용량·정책 검토 후 instance에 실제 제출하는 요청의 도착률 | NLR 설정 request rate와 의미가 일치하는지 확인한 뒤 M3+R3 입력으로 연결 |
| Rejected / unadmitted rate | 제출하지 못한 요청률 | 미처리 지표로 별도 기록. 입력에서 조용히 삭제하지 않음 |
| Completed throughput | 실제 응답을 완료한 요청률 | queue·처리시간·token 길이 등의 결과. 별도 모델 없이 admitted rate와 같다고 단정하지 않음 |

- NLR M3+R3가 학습한 것은 benchmark의 설정 도착률에 따른 전력 관계임. 실제 완료량에 대한 전력식으로 바꾸어 설명하지 않음.
- 간단한 admission 모델을 사용할 때에는 `offered = admitted + unadmitted`라는 요청 보존 관계를 확인함. 이는 운영 입력의 회계 관계이며, queueing 성능 모델을 의미하지 않음.
- 한도를 넘는 요청을 계산상 잘랐다고 해서 latency나 실제 완료율을 검증한 것은 아님.
- 기존 실험의 `burstiness=100000` 및 prompt 조건과 매우 다른 도착 패턴·token 분포를 적용하면, 동일한 평균 request rate라도 전력 특성이 같다는 보장은 없음. 이 차이는 외삽 가정으로 명시함.
- [3] §2.3.2의 inference 사례는 latency에 대응하는 rate 한도와 미처리 요청을 사용하며, §2.3.1의 job FIFO 대기열을 그대로 적용한 구조가 아님.*

## 4. 보존할 독립 합산 기준선

기준선 경로: [heterogeneous_equivalent_facility_17mw](D:/DC_modeling/aidc-dfl-doe-research/outputs/heterogeneous_equivalent_facility_17mw)

| 파일 | 용도 |
|---|---|
| [generation_config.json](D:/DC_modeling/aidc-dfl-doe-research/outputs/heterogeneous_equivalent_facility_17mw/generation_config.json) | 단위 수, workload 구성비, rate 범위, 시간축, seed 및 scaling 여부 |
| [scenario_summary.csv](D:/DC_modeling/aidc-dfl-doe-research/outputs/heterogeneous_equivalent_facility_17mw/scenario_summary.csv) | 각 scenario의 평균·표준편차·ramp 등 비교 기준 |
| [sampled_conditions.csv](D:/DC_modeling/aidc-dfl-doe-research/outputs/heterogeneous_equivalent_facility_17mw/sampled_conditions.csv) | scenario별로 샘플링한 workload 조건의 요약 |
| [facility_profiles_long.csv](D:/DC_modeling/aidc-dfl-doe-research/outputs/heterogeneous_equivalent_facility_17mw/facility_profiles_long.csv) | 시설 합산 및 서비스별 시간경로 |
| [report_ko.md](D:/DC_modeling/aidc-dfl-doe-research/outputs/heterogeneous_equivalent_facility_17mw/report_ko.md:27) | 기존 통계 요약의 출처 |

- 6,487 equivalent workload units: InstructCoder 1,946개, Llama2-data 4,541개.
- 30/70은 **unit 수 구성비**이며, request 구성비나 정확한 전력 구성비를 뜻하지 않음.
- unit별 request rate는 scenario 시작 시 한 번 샘플링하고 180초 동안 고정함. residual은 unit별 독립으로 생성함.
- 0.1초 간격, 30개 facility scenario. 서로 다른 scenario에서는 rate 샘플과 residual 실현이 달라짐.
- 목표 IT 평균은 `20 MW / 1.15 = 17.391304 MW`; 합산 후 목표 평균을 맞추는 배율 조정은 하지 않음.
- 구현 근거: [rate 샘플링 및 합산](D:/DC_modeling/aidc-dfl-doe-research/src/nlr_data/heterogeneous_facility_aggregation.py:166), [scenario별 seed](D:/DC_modeling/aidc-dfl-doe-research/src/nlr_data/heterogeneous_facility_aggregation.py:260).

아래 수치는 **기존 report에 기록된 값**을 인용한 것이며, 이번 감사에서 실험을 재실행하거나 통계를 독립 재산출한 값이 아님.

| 지표 | 기존 결과 |
|---|---:|
| 30개 scenario 평균전력의 평균 | 17.391110 MW |
| 각 scenario 내 시간축 표준편차의 평균 | 0.006607 MW = 6.607 kW |
| 0.1초 절대 ramp P95의 scenario 평균 | 0.031378 MW/s |
| 1초 절대 ramp P95의 scenario 평균 | 0.009256 MW/s |

**판단:** 고정 운전점에서 독립 residual을 합산한 다양성 기준선으로 보존함. 변동폭이 작다는 사실만으로 잘못된 모델이라고 판정하지 않으며, 시간가변 시설 수요 전체를 재현했다고도 주장하지 않음.

## 5. 현재 M3+R3 parameter와 API

- 로드: [FinalITProfileGenerator.from_benchmark_artifact](D:/DC_modeling/aidc-dfl-doe-research/src/nlr_data/final_it_profile_generator.py:429).
- artifact 위치: [r3_high_rate_benchmark](D:/DC_modeling/aidc-dfl-doe-research/outputs/r3_high_rate_benchmark).
- 필요 파일: `train_validation_m3_parameters.csv`, `train_validation_residual_parameters.csv`, `protocol.json`, `status.json`.
- M3 조회: `generator.m3_parameters[prompt_dataset].predict(request_rate)`.
- R3 조회: `generator.residual_library.weighted(prompt_dataset, request_rate, column)`.
- R3 보정표는 prompt dataset별 80개 조건, rate 1~80 request/s의 train+validation으로 구성됨. 81~100 request/s는 이미 과거에 노출된 high-rate benchmark이며 미사용 최종 test라고 부르지 않음. 근거: [보정표](D:/DC_modeling/aidc-dfl-doe-research/outputs/r3_high_rate_benchmark/train_validation_residual_parameters.csv), [protocol의 exposure 기록](D:/DC_modeling/aidc-dfl-doe-research/outputs/r3_high_rate_benchmark/protocol.json:9).
- R3 조건 조회는 동일 dataset 안에서 rate가 가까운 8개 행을 선택하고 `1/max(rate 차이, 0.5)` 가중치를 정규화해 적용함. 이는 현재 구현의 parameter 보간 규칙이며 [3]의 원래 전력식이 아님. 근거: [ResidualParameterLibrary](D:/DC_modeling/aidc-dfl-doe-research/src/nlr_data/residual_model_selection.py:71).
- 주요 R3 항목: `phi`, `innovation_std_kw`, `jump_evidence`, negative/background/positive의 weight·mean·standard deviation.
- 현재 생성기는 0.1초 보정을 전제로 함. sample interval을 임의로 바꾸면 자기상관과 innovation 통계의 의미가 달라짐.

## 6. 시간가변 연결 시 필요한 최소 보완

### 6.1 기존 generator를 구간별로 반복 호출하지 않음

기존 [simulate_residual_ensemble](D:/DC_modeling/aidc-dfl-doe-research/src/nlr_data/residual_model_selection.py:261)은 한 번 호출할 때 하나의 고정 request rate를 사용하고, [출력 마지막 단계](D:/DC_modeling/aidc-dfl-doe-research/src/nlr_data/residual_model_selection.py:367)에서 각 유한 구간의 residual 평균을 뺌.

- traffic 구간마다 이 함수를 새로 실행하면 residual 상태가 이어지지 않음.
- 각 구간 평균을 다시 0으로 맞추면 구간 간 평균·에너지 변동을 제거할 수 있음.
- 단, **기존 전체 180초 residual에서 상수를 한 번 빼는 연산 자체는 그 구간 안의 ramp와 시간축 표준편차를 바꾸지 않음**. 현재 작은 ramp의 직접 원인이라고 설명하면 안 됨.

후속 구현은 기존 함수를 변경하지 않고 별도 stateful 경로로 구성하는 것이 안전함.

- 직전 residual 상태를 다음 구간으로 전달함.
- 조건이 바뀔 때 phi 및 innovation parameter의 적용 규칙을 명시함.
- 구간별 유한 표본 평균 제거와 목표 평균 강제 정규화를 하지 않음.
- innovation의 기대값을 0으로 맞추는 parameter 처리와, 생성된 유한 표본의 평균을 사후 제거하는 처리를 구분함.
- 구간 경계의 전력 점프가 수요 변화인지, 정상상태 M3의 즉시응답 가정인지, residual 재시작 때문인지 분리해 확인함.

### 6.2 아직 검증하지 않은 부분

- 시간가변 `r(t)`를 M3에 넣는 것은 고정 운전점 식의 quasi-steady 적용임. 실제 부하의 운전점 전이시간이나 queue dynamics를 식별한 결과가 아님.
- request rate가 보정·benchmark 범위를 벗어나면 표시해야 함. 특히 무요청·idle 구간은 `M3(0)`로 자동 대체하지 않음.
- warm instance의 전력과 NLR idle 시험값의 동일성은 별도 확인 대상임.
- 180초 보정에서 하루 등 장시간으로 확장하면 조건부 residual 통계가 장시간에도 유지된다는 추가 가정을 사용하게 됨.
- 실제 traffic의 시간축을 보존해야 하며, 시간·일 단위 패턴을 180초로 압축해 고속 ramp를 만들지 않음.
- 기존 독립 residual은 유지할 수 있음. 단순히 변동폭을 키우기 위한 임의 상관·동기화는 추가하지 않음.
- [3]은 0.1초 node power를 측정했지만, 시설 inference 결과는 1분 timestep으로 제시함. 따라서 [3]을 0.1초 시설 ramp의 검증 근거로 사용하지 않음.*

## 7. 후속 결과의 비교 기준

- 기존 독립 합산안은 역사적 기준선으로 남김.
- 새 분배 규칙 안에서 동일한 instance/node 수, 서비스 구성, 분배 정책, 평균 offered traffic을 사용해 고정 traffic과 시간가변 traffic을 비교함.
- 이 대조군이 있어야 시간가변 효과와 분배 정책 변경 효과를 분리할 수 있음.
- 시설 규모나 traffic 수준은 실험 전에 결정하고, 각 출력의 평균을 사후 17.39 MW로 강제 맞추지 않음.
- 새로운 결과에 더 큰 ramp가 생겨도 그것만으로 현실성이 향상됐다고 판단하지 않음. 같은 전력 경계·해상도의 관측 또는 문헌 통계와 비교함.
- 측정 경계와 고속 전이응답이 검증되기 전에는 결과를 **NLR 전력자료에 기반한 시설 IT 시나리오**로 보고하며 실제 AIDC PCC 파형의 재현이라고 표현하지 않음.

## Reference 각주 및 차용 경계

\* **[3]** Roberto Vercellino et al., *Measurement of Generative AI Workload Power Profiles for Whole-Facility Data Center Infrastructure Planning*, arXiv:2604.07345v1, 2026. [원문](https://arxiv.org/html/2604.07345v1), [로컬 PDF](<D:/DC_modeling/outputs/literature_review_example/paper_list/[3] Measurement of Generative AI Workload Power Profiles for Whole-Facility Data Center Infrastructure Planning.pdf>).

- §2.2.1: WattAMeter를 통한 CPU/GPU power 측정.
- §2.3.2, Fig.3, 식(2)~(7): 시간별 inference 요청, latency에 따른 instance당 rate 한도, instance 분배 및 전력 sample 할당.
- §3.1: Kestrel node 구성인 H100 GPU 4개 및 AMD EPYC CPU 2개.
- §3.5, Fig.13: 지속된 request rate 조건에서 3분 power sample을 수집하여 시설 모형에 사용.
- §4.2, Fig.15: 한 instance를 한 node에 배치한 사례 및 1분 시간간격의 시설 시뮬레이션.
- Conclusions: 냉각·전력분배 손실 등 비IT 부하를 시설 시뮬레이션에서 제외했다는 한계.

이 문서에서 **M3+R3로 실측 sample을 대체하는 것, parameter 보간, 시간가변 stateful 확장 및 합산 residual 독립성은 본 연구의 구현·가정**임. [3]이 해당 수식이나 0.1초 시설 전이모델을 그대로 제공한다고 주장하지 않음. M3·R3 개별 구조의 기존 문헌 연결은 [고정 생성기 method references](D:/DC_modeling/aidc-dfl-doe-research/src/nlr_data/final_it_profile_generator.py:6) 및 [benchmark protocol](D:/DC_modeling/aidc-dfl-doe-research/outputs/r3_high_rate_benchmark/protocol.json)에 기록되어 있으며, 이 감사는 그 문헌들의 원문 수식 전체를 재검증한 작업이 아님.
