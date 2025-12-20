"""optimized_autotuning.py - Compiled variant for the autotuning benchmark.

Pairs with `baseline_autotuning.py`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

import ch20.arch_config  # noqa: F401 - Apply chapter defaults


from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.utils.compile_utils import compile_model


class AutotuneModel(nn.Module):
    """Pointwise-heavy block that benefits from compiler fusion."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 0.01
        y = y * self.scale + self.bias
        y = torch.nn.functional.silu(y)
        y = (y * 1.0001) + 0.0001
        y = y * 0.999 + 0.001
        y = torch.nn.functional.silu(y)
        y = (y * 1.0001) + 0.0001
        y = y * 0.999 + 0.001
        y = torch.nn.functional.silu(y)
        y = (y * 1.0001) + 0.0001
        y = y * 0.999 + 0.001
        y = torch.nn.functional.silu(y)
        y = (y * 1.0001) + 0.0001
        y = y * 0.999 + 0.001
        return y


class OptimizedAutotuningBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Runs a small model with torch.compile to validate autotune plumbing."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.batch = 512
        self.hidden_dim = 4096
        tokens = self.batch * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self._verify_input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = AutotuneModel(self.hidden_dim).to(self.device, dtype=torch.bfloat16).eval()
        # Compile once in setup; benchmark_fn measures steady-state execution.
        self.model = compile_model(
            model,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
            dynamic=False,
        )
        self.inputs = torch.randn(self.batch, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
        self._verify_input = self.inputs[0:1].clone()
        # Warm enough runs to ensure autotuning completes before timing.
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(self.inputs)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.model is not None and self.inputs is not None
        with self._nvtx_range("optimized_autotuning"):
            with torch.no_grad():
                _ = self.model(self.inputs)
            self._synchronize()

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.model is None:
            raise RuntimeError("setup() must prepare verify input before verification")
        with torch.no_grad():
            self.output = self.model(self._verify_input).float().clone()
        self._set_verification_payload(
            inputs={"verify_input": self._verify_input},
            output=self.output,
            batch_size=int(self._verify_input.shape[0]),
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=10,
            use_subprocess=True,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.inputs is None:
            return "Input tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedAutotuningBenchmark()
