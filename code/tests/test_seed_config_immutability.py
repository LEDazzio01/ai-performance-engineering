"""Regression tests for harness immutability guarantees.

These tests ensure benchmarks cannot mutate harness config at runtime and cannot
reseed RNGs during perf runs.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    LaunchVia,
    TorchrunLaunchSpec,
)


class _CpuBenchmarkBase(BaseBenchmark):
    allow_cpu = True

    def __init__(self) -> None:
        super().__init__()
        # Force CPU execution even when CUDA is available in the test environment.
        self.device = torch.device("cpu")
        self.input_tensor: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

    def validate_result(self) -> Optional[str]:
        return None

    def get_verify_inputs(self) -> dict:
        if self.input_tensor is None:
            raise RuntimeError("setup() must set input_tensor")
        return {"input": self.input_tensor}

    def get_verify_output(self) -> torch.Tensor:
        if self.output is None:
            raise RuntimeError("benchmark_fn() must set output")
        return self.output

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def get_input_signature(self) -> dict:
        if self.input_tensor is None:
            raise RuntimeError("setup() must be called before get_input_signature()")
        return {"shape": tuple(self.input_tensor.shape), "dtype": str(self.input_tensor.dtype)}


class MutateConfigBenchmark(_CpuBenchmarkBase):
    def setup(self) -> None:
        cfg = self.get_config()
        if cfg is None:
            raise RuntimeError("Missing config in setup()")
        # Benchmarks must not mutate harness config; this should fail fast.
        cfg.iterations = 1  # type: ignore[attr-defined]

    def benchmark_fn(self) -> None:
        self.input_tensor = torch.zeros(8, device=self.device)
        self.output = self.input_tensor


class MutateSeedBenchmark(_CpuBenchmarkBase):
    def setup(self) -> None:
        torch.manual_seed(123)
        self.input_tensor = torch.randn(8, device=self.device)

    def benchmark_fn(self) -> None:
        self.output = self.input_tensor + 1


def _make_cpu_harness() -> BenchmarkHarness:
    config = BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=2,
        warmup=5,
        enable_profiling=False,
        enable_memory_tracking=False,
        use_subprocess=False,
    )
    return BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)


def test_benchmark_config_is_read_only_during_run():
    harness = _make_cpu_harness()
    result = harness.benchmark(MutateConfigBenchmark())
    assert result.errors, "Expected config mutation to fail"
    assert any("read-only" in err.lower() for err in result.errors)


def test_seed_mutation_is_detected_in_perf_runs():
    harness = _make_cpu_harness()
    result = harness.benchmark(MutateSeedBenchmark())
    assert result.errors, "Expected seed mutation to fail"
    assert any("seed mutation detected" in err.lower() for err in result.errors)


class TorchrunSeedMutateBenchmark(_CpuBenchmarkBase):
    def __init__(self, script_path: Path) -> None:
        super().__init__()
        self._script_path = script_path

    def setup(self) -> None:  # pragma: no cover - not executed in torchrun mode
        self.input_tensor = torch.zeros(1, device=self.device)

    def benchmark_fn(self) -> None:  # pragma: no cover - not executed in torchrun mode
        if self.input_tensor is None:
            raise RuntimeError("setup() must set input_tensor")
        self.output = self.input_tensor

    def get_torchrun_spec(self, config: BenchmarkConfig) -> TorchrunLaunchSpec:
        return TorchrunLaunchSpec(
            script_path=self._script_path,
            script_args=[],
            env={},
            parse_rank0_only=True,
            multi_gpu_required=False,
            name="torchrun_seed_mutate",
            config_arg_map={},
        )


def test_seed_mutation_is_detected_in_torchrun_runs(tmp_path: Path):
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not available in PATH")

    script = tmp_path / "seed_mutate_script.py"
    script.write_text(
        "import torch\n"
        "torch.manual_seed(123)\n"
        "print('done')\n",
        encoding="utf-8",
    )

    config = BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=1,
        warmup=1,
        enable_profiling=False,
        enable_memory_tracking=False,
        use_subprocess=False,
        launch_via=LaunchVia.TORCHRUN,
        nproc_per_node=1,
        multi_gpu_required=False,
    )
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(TorchrunSeedMutateBenchmark(script))

    assert result.errors, "Expected torchrun seed mutation to fail"
    joined = "\n".join(result.errors).lower()
    assert "seed mutation detected" in joined
