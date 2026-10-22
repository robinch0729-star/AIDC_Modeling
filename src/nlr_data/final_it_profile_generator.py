"""Portable final M3+R3 IT-load profile generator.

The generator consumes the locked artifacts produced by
``r3_high_rate_benchmark.py``.  It does not refit or reselect a model.

Method references
-----------------
* M3: the request-rate saturation observed in NLR Paper list [3]; the exact
  prompt-conditioned logistic equation is this study's implementation.
* Mean plus residual and sampled AR(1): adapted from Paper list [37].
* R3 Jump-OU backbone: modified from Paper list [2], Eq. (2).  The signed
  Gaussian-mixture innovation is an NLR-specific approximation in this study.

Scientific boundary
-------------------
The source NLR traces are job-total IT power, not a documented whole-facility
PCC load or one physical server.  Optional MW scaling is therefore an
equivalent normalized IT-load scenario, not a claim about server count,
cooling demand, UPS loss, or PCC power.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import savemat

from .dynamic_identifiability import M3Parameters
from .residual_model_selection import (
    ResidualParameterLibrary,
    simulate_residual_ensemble,
)


FIXED_RESIDUAL_MODEL = "R3_jump_ou"
CALIBRATED_INTERVAL_S = 0.1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GenerationRequest:
    prompt_dataset: str
    request_rate: float
    duration_s: float = 180.0
    interval_s: float = CALIBRATED_INTERVAL_S
    scenario_count: int = 100
    seed: int = 42
    target_it_mean_mw: float | None = None
    allow_unsupported_extrapolation: bool = False


@dataclass
class FinalITProfileGenerator:
    """Locked M3+R3 generator loaded from a benchmark artifact directory."""

    m3_parameters: dict[str, M3Parameters]
    residual_library: ResidualParameterLibrary
    source_dir: Path
    protocol: dict[str, Any]
    benchmark_status: dict[str, Any]

    @property
    def prompt_datasets(self) -> tuple[str, ...]:
        return tuple(sorted(self.m3_parameters))

    @property
    def calibrated_interval_s(self) -> float:
        intervals = pd.to_numeric(
            self.residual_library.parameters["interval_s"], errors="coerce"
        ).dropna()
        if intervals.empty:
            return CALIBRATED_INTERVAL_S
        value = float(intervals.median())
        if not np.allclose(intervals.to_numpy(float), value, rtol=0.0, atol=1e-8):
            raise ValueError("Residual parameter artifact contains multiple sampling intervals")
        return value

    def rate_ranges(self, prompt_dataset: str) -> dict[str, tuple[float, float]]:
        if prompt_dataset not in self.m3_parameters:
            raise KeyError(
                f"Unknown prompt_dataset={prompt_dataset!r}; "
                f"available={self.prompt_datasets}"
            )
        part = self.residual_library.parameters.loc[
            self.residual_library.parameters["prompt_dataset"].eq(prompt_dataset)
        ]
        calibration = (
            float(part["request_rate"].min()),
            float(part["request_rate"].max()),
        )
        benchmark_range = self.protocol.get("request_rate_range", [np.nan, np.nan])
        benchmark = (float(benchmark_range[0]), float(benchmark_range[1]))
        return {"calibration": calibration, "high_rate_benchmark": benchmark}

    def request_rate_status(self, prompt_dataset: str, request_rate: float) -> str:
        ranges = self.rate_ranges(prompt_dataset)
        low, high = ranges["calibration"]
        bench_low, bench_high = ranges["high_rate_benchmark"]
        if low <= request_rate <= high:
            return "train_validation_interpolation"
        if bench_low <= request_rate <= bench_high:
            return "validated_high_rate_extrapolation"
        return "unsupported_extrapolation"

    def validate_request(self, request: GenerationRequest) -> str:
        if request.prompt_dataset not in self.m3_parameters:
            raise KeyError(
                f"Unknown prompt_dataset={request.prompt_dataset!r}; "
                f"available={self.prompt_datasets}"
            )
        if not np.isfinite(request.request_rate) or request.request_rate < 0:
            raise ValueError("request_rate must be finite and nonnegative")
        if not np.isfinite(request.duration_s) or request.duration_s <= 0:
            raise ValueError("duration_s must be finite and positive")
        if request.scenario_count < 1:
            raise ValueError("scenario_count must be at least one")
        if not np.isclose(
            request.interval_s,
            self.calibrated_interval_s,
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError(
                f"R3 was calibrated at dt={self.calibrated_interval_s:g} s; "
                "arbitrary resampling would alter phi and innovation statistics"
            )
        if request.target_it_mean_mw is not None:
            if (
                not np.isfinite(request.target_it_mean_mw)
                or request.target_it_mean_mw <= 0
            ):
                raise ValueError("target_it_mean_mw must be finite and positive")
        status = self.request_rate_status(
            request.prompt_dataset, float(request.request_rate)
        )
        if status == "unsupported_extrapolation" and not request.allow_unsupported_extrapolation:
            ranges = self.rate_ranges(request.prompt_dataset)
            raise ValueError(
                f"request_rate={request.request_rate:g} request/s is outside the "
                f"calibration range {ranges['calibration']} and evaluated high-rate "
                f"range {ranges['high_rate_benchmark']}; explicitly enable unsupported "
                "extrapolation to proceed"
            )
        return status

    def generate(self, request: GenerationRequest) -> pd.DataFrame:
        """Generate long-form equivalent IT-power scenarios.

        The R3 residual is zero-centered per finite scenario, so the raw
        scenario mean equals the M3 operating point.  If ``target_it_mean_mw``
        is supplied, one deterministic scale factor maps that operating point
        to the requested equivalent facility mean.
        """
        rate_status = self.validate_request(request)
        length = int(round(request.duration_s / request.interval_s))
        if length < 2:
            raise ValueError("duration_s must contain at least two samples")

        m3 = self.m3_parameters[request.prompt_dataset]
        steady_kw = float(m3.predict(request.request_rate))
        residual = simulate_residual_ensemble(
            FIXED_RESIDUAL_MODEL,
            draws=request.scenario_count,
            length=length,
            prompt_dataset=request.prompt_dataset,
            request_rate=float(request.request_rate),
            library=self.residual_library,
            rng=np.random.default_rng(request.seed),
        )
        power_kw = steady_kw + residual
        if float(np.min(power_kw)) < 0.0:
            raise RuntimeError(
                "Generated negative IT power; the requested condition is outside the "
                "physically usable range of the calibrated additive model"
            )

        scenario_id = np.repeat(np.arange(request.scenario_count, dtype=int), length)
        sample_index = np.tile(np.arange(length, dtype=int), request.scenario_count)
        time_s = sample_index.astype(float) * request.interval_s
        frame = pd.DataFrame(
            {
                "scenario_id": scenario_id,
                "sample_index": sample_index,
                "time_s": time_s,
                "prompt_dataset": request.prompt_dataset,
                "request_rate": float(request.request_rate),
                "request_rate_status": rate_status,
                "master_seed": int(request.seed),
                "residual_model": FIXED_RESIDUAL_MODEL,
                "m3_steady_power_kw": steady_kw,
                "residual_kw": residual.reshape(-1),
                "it_power_kw": power_kw.reshape(-1),
            }
        )
        if request.target_it_mean_mw is not None:
            scale_mw_per_kw = float(request.target_it_mean_mw) / steady_kw
            frame["equivalent_scale_mw_per_kw"] = scale_mw_per_kw
            frame["target_it_mean_mw"] = float(request.target_it_mean_mw)
            frame["it_power_mw"] = frame["it_power_kw"] * scale_mw_per_kw
        return frame

    def save_generation(
        self,
        frame: pd.DataFrame,
        request: GenerationRequest,
        output_dir: str | Path,
    ) -> Path:
        """Save CSV, MATLAB MAT, figure, configuration, and method report."""
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        expected_rows = request.scenario_count * int(
            round(request.duration_s / request.interval_s)
        )
        if len(frame) != expected_rows:
            raise ValueError("frame does not match the supplied generation request")

        ordered = frame.sort_values(["scenario_id", "sample_index"], kind="stable")
        count = int(ordered["scenario_id"].nunique())
        length = int(ordered["sample_index"].nunique())
        time_s = ordered.loc[ordered["scenario_id"].eq(0), "time_s"].to_numpy(float)
        residual_kw = ordered["residual_kw"].to_numpy(float).reshape(count, length)
        power_kw = ordered["it_power_kw"].to_numpy(float).reshape(count, length)

        ordered.to_csv(output / "it_profiles_long.csv", index=False, encoding="utf-8-sig")
        wide = pd.DataFrame(
            {
                "time_s": time_s,
                **{
                    f"scenario_{scenario:03d}_it_power_kw": power_kw[scenario]
                    for scenario in range(count)
                },
            }
        )
        wide.to_csv(output / "it_profiles_wide.csv", index=False, encoding="utf-8-sig")

        mat_payload: dict[str, Any] = {
            "time_s": time_s,
            "it_power_kw": power_kw,
            "residual_kw": residual_kw,
            "m3_steady_power_kw": float(ordered["m3_steady_power_kw"].iloc[0]),
            "request_rate": float(request.request_rate),
            "scenario_id": np.arange(count, dtype=int),
            "master_seed": int(request.seed),
        }
        if "it_power_mw" in ordered:
            power_mw = ordered["it_power_mw"].to_numpy(float).reshape(count, length)
            mat_payload["it_power_mw"] = power_mw
            mat_payload["target_it_mean_mw"] = float(request.target_it_mean_mw)
            wide_mw = pd.DataFrame(
                {
                    "time_s": time_s,
                    **{
                        f"scenario_{scenario:03d}_it_power_mw": power_mw[scenario]
                        for scenario in range(count)
                    },
                }
            )
            wide_mw.to_csv(
                output / "it_profiles_wide_mw.csv", index=False, encoding="utf-8-sig"
            )
        savemat(output / "it_profiles.mat", mat_payload, do_compression=True)

        source_files = {
            "m3_parameters": self.source_dir / "train_validation_m3_parameters.csv",
            "r3_parameters": self.source_dir / "train_validation_residual_parameters.csv",
            "protocol": self.source_dir / "protocol.json",
            "benchmark_status": self.source_dir / "status.json",
        }
        for name, source in source_files.items():
            shutil.copy2(source, output / f"source_{source.name}")

        configuration = {
            "artifact_type": "locked M3+R3 equivalent IT profile generation",
            "artifact_version": 1,
            "request": asdict(request),
            "request_rate_status": str(ordered["request_rate_status"].iloc[0]),
            "fixed_residual_model": FIXED_RESIDUAL_MODEL,
            "model_reselected": False,
            "m3_parameters": {
                name: asdict(parameter)
                for name, parameter in self.m3_parameters.items()
            },
            "source_artifacts": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in source_files.items()
            },
            "measurement_scope": "NLR job-total IT power",
            "pcc_load_claimed": False,
            "cooling_ups_auxiliary_included": False,
            "mw_scaling_interpretation": (
                "equivalent normalized IT load; not physical server aggregation"
                if request.target_it_mean_mw is not None
                else None
            ),
            "references": {
                "M3*": "NLR Paper list [3] saturation; exact logistic is this study",
                "mean_residual_AR1**": "Paper list [37], modified application",
                "R3_Jump_OU***": "Paper list [2] Eq. (2), signed-GMM modification",
            },
        }
        (output / "generation_config.json").write_text(
            json.dumps(configuration, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        self._save_plot(ordered, output / "generated_it_profiles.png")
        (output / "generation_report_ko.md").write_text(
            self._method_report(ordered, request), encoding="utf-8"
        )
        return output

    def _save_plot(self, frame: pd.DataFrame, output: Path) -> None:
        power_column = "it_power_mw" if "it_power_mw" in frame else "it_power_kw"
        unit = "MW" if power_column.endswith("_mw") else "kW"
        figure, axis = plt.subplots(figsize=(12, 5), dpi=160)
        for scenario_id, group in frame.groupby("scenario_id", observed=True):
            if int(scenario_id) >= 10:
                break
            axis.plot(
                group["time_s"],
                group[power_column],
                lw=0.8,
                alpha=0.7,
                label=f"scenario {int(scenario_id)}" if int(scenario_id) < 5 else None,
            )
        steady = (
            float(frame["target_it_mean_mw"].iloc[0])
            if power_column == "it_power_mw"
            else float(frame["m3_steady_power_kw"].iloc[0])
        )
        axis.axhline(steady, color="black", ls="--", lw=1.2, label="M3 operating point")
        axis.set(
            title=(
                f"Final M3+R3 IT profiles | {frame['prompt_dataset'].iloc[0]} | "
                f"{frame['request_rate'].iloc[0]:g} request/s"
            ),
            xlabel="Time (s)",
            ylabel=f"IT power ({unit})",
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=3, loc="upper right")
        figure.tight_layout()
        figure.savefig(output, bbox_inches="tight")
        plt.close(figure)

    def _method_report(self, frame: pd.DataFrame, request: GenerationRequest) -> str:
        power_column = "it_power_mw" if "it_power_mw" in frame else "it_power_kw"
        unit = "MW" if power_column.endswith("_mw") else "kW"
        grouped = frame.groupby("scenario_id", observed=True)[power_column]
        summary = pd.DataFrame(
            {
                "mean": grouped.mean(),
                "minimum": grouped.min(),
                "maximum": grouped.max(),
                "std": grouped.std(ddof=0),
            }
        )
        ranges = self.rate_ranges(request.prompt_dataset)
        return rf"""# 최종 M3+R3 IT 부하 프로파일 생성 결과

## 1. 생성식

\[
P_{{IT,k}}=P_{{M3}}(r,d)+e_k \quad *
\]

\[
e_{{k+1}}=\phi(r,d)e_k+\eta_k \quad **
\]

\[
\eta_k\sim(1-g)\mathcal{{N}}(0,\sigma^2)
+g\sum_c\pi_c\mathcal{{N}}(\mu_c,\sigma_c^2) \quad ***
\]

- *: NLR Paper list [3]의 request-rate 포화 특성을 반영한 본 연구의 prompt-conditioned M3 logistic.
- **: Paper list [37]의 mean-template+AR(1) residual 구조를 수정 적용.
- ***: Paper list [2] 식 (2)의 OU+Jump를 NLR 양·음 변동용 signed GMM innovation으로 수정 적용.

## 2. 입력 조건

- Prompt dataset: `{request.prompt_dataset}`
- Request rate: {request.request_rate:g} request/s
- Request-rate 상태: `{frame['request_rate_status'].iloc[0]}`
- Train+Validation 보정 범위: {ranges['calibration'][0]:g}-{ranges['calibration'][1]:g} request/s
- 고율 benchmark 범위: {ranges['high_rate_benchmark'][0]:g}-{ranges['high_rate_benchmark'][1]:g} request/s
- Scenario 수: {request.scenario_count}
- Duration: {request.duration_s:g} s
- dt: {request.interval_s:g} s
- Master seed: {request.seed}
- 출력 단위: {unit}

## 3. 생성 결과 요약

```text
{summary.describe().round(6).to_string()}
```

![Generated profiles](generated_it_profiles.png)

## 4. 적용 경계

- 생성기는 benchmark 이후 모델을 다시 선택하거나 파라미터를 재학습하지 않는다.
- dt=0.1초에서 추정된 R3 파라미터를 사용하므로 현재 구현은 dt=0.1초만 허용한다.
- 81-100 request/s는 Train+Validation 보간이 아니라 검증된 고율 외삽 구간이다.
- NLR 원신호는 job-total IT power이며, 냉각·UPS 손실·보조부하는 포함하지 않는다.
- MW 출력은 평균 크기를 맞춘 equivalent IT scaling이며 실제 서버 수 또는 PCC 부하를 의미하지 않는다.
- 현재 입력은 일정한 request rate 운전조건이다. 시간변화 request-rate에 대한 인과적 과도응답은 NLR 데이터만으로 식별되지 않았다.
"""

    @classmethod
    def from_benchmark_artifact(
        cls, artifact_dir: str | Path
    ) -> "FinalITProfileGenerator":
        source = Path(artifact_dir).expanduser().resolve()
        required = {
            "m3": source / "train_validation_m3_parameters.csv",
            "r3": source / "train_validation_residual_parameters.csv",
            "protocol": source / "protocol.json",
            "status": source / "status.json",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing benchmark artifacts: {missing}")

        protocol = json.loads(required["protocol"].read_text(encoding="utf-8"))
        status = json.loads(required["status"].read_text(encoding="utf-8"))
        if protocol.get("fixed_model_before_benchmark") != FIXED_RESIDUAL_MODEL:
            raise ValueError("Protocol does not lock R3_jump_ou")
        if not bool(protocol.get("selection_locked")):
            raise ValueError("Protocol did not lock model selection")
        if status.get("fixed_model") != FIXED_RESIDUAL_MODEL:
            raise ValueError("Benchmark status does not identify fixed R3")
        if bool(status.get("model_reselected")):
            raise ValueError("Benchmark artifact reports post-benchmark reselection")

        m3_table = pd.read_csv(required["m3"])
        expected_m3 = {"prompt_dataset", "p0_kw", "delta_p_kw", "k", "x0"}
        if not expected_m3.issubset(m3_table.columns):
            raise ValueError(f"M3 table is missing {sorted(expected_m3-m3_table.columns)}")
        m3 = {
            str(row.prompt_dataset): M3Parameters(
                p0_kw=float(row.p0_kw),
                delta_p_kw=float(row.delta_p_kw),
                k=float(row.k),
                x0=float(row.x0),
            )
            for row in m3_table.itertuples(index=False)
        }

        residual_parameters = pd.read_csv(required["r3"])
        required_residual = {
            "prompt_dataset",
            "request_rate",
            "interval_s",
            "residual_std_kw",
            "phi",
            "innovation_std_kw",
            "jump_evidence",
            "negative_weight",
            "negative_mean_kw",
            "negative_std_kw",
            "background_weight",
            "background_mean_kw",
            "background_std_kw",
            "positive_weight",
            "positive_mean_kw",
            "positive_std_kw",
        }
        missing_residual = required_residual.difference(residual_parameters.columns)
        if missing_residual:
            raise ValueError(
                f"R3 parameter table is missing {sorted(missing_residual)}"
            )
        if set(residual_parameters["split"].dropna().astype(str)) != {
            "train",
            "validation",
        }:
            raise ValueError("Expected exactly Train+Validation R3 calibration rows")

        generator = cls(
            m3_parameters=m3,
            residual_library=ResidualParameterLibrary(residual_parameters, {}),
            source_dir=source,
            protocol=protocol,
            benchmark_status=status,
        )
        # Eager validation prevents creating an apparently usable bad artifact.
        _ = generator.calibrated_interval_s
        if set(generator.prompt_datasets) != set(
            residual_parameters["prompt_dataset"].astype(str).unique()
        ):
            raise ValueError("M3 and R3 prompt-dataset sets do not match")
        return generator
