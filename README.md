# AIDC DFL–DOE Research

배전계통에 연계된 AI 데이터센터(AIDC)의 부하·불확실성 정보를 추정하고, 이를 DSO의 Dynamic Operating Envelope(DOE) 산정과 ESS–TES 기반 local dispatch에 활용하기 위한 연구 코드 저장소입니다.

## 연구 프레임워크

```text
AIDC gray-box 모델 및 NLR workload
→ DFL 기반 부하·불확실성 정보추정
→ AIDC가 추정정보를 DSO에 전달
→ DSO OPF 기반 PCC import DOE 할당
→ Energy Router의 ESS–TES local dispatch
→ OPAL-RT 기반 DOE 지령추종 검증
```

현재 우선 수행 범위는 **NLR 데이터셋 탐색 및 workload 특성추출 파이프라인 구축**입니다.

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

## 초기 개발 계획

1. NLR 폴더·파일 메타데이터 인덱싱
2. 스키마, 시간축, 단위 및 결측치 점검
3. workload 유형별 시계열 정규화
4. 평균·최댓값·분산·ramp rate·지속시간·주기성 특성 추출
5. 특성 테이블 및 품질관리 보고서 생성
6. gray-box 보정과 DFL 입력으로 사용할 데이터셋 구성

## Git 운용 원칙

- 원본 NLR 데이터, 가상환경, 모델 checkpoint 및 자동 생성 결과는 커밋하지 않습니다.
- 코드, 설정 예시, 작은 검증용 샘플 및 재현 가능한 분석 절차만 버전 관리합니다.
- 기능 단위 브랜치와 명확한 commit message를 사용합니다.

첫 개발 브랜치 예시:

```powershell
git switch -c feature/nlr-feature-extraction
```

