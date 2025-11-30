"""baseline_triton.py - Baseline Triton matmul wrapper."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402
try:
    import triton  # noqa: F401
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


def baseline_elementwise(input_tensor: torch.Tensor, output: torch.Tensor, **_: int) -> None:
    """
    Baseline element-wise operation using standard PyTorch.
    This is compared against Triton's optimized kernel in the optimized version.
    """
    # Standard PyTorch operation: output = input * 2.0 + 1.0
    output.copy_(input_tensor * 2.0 + 1.0)


class BaselineTritonBenchmark(BaseBenchmark):
    """Baseline element-wise operation using standard PyTorch (compared to Triton kernel)."""

    def __init__(self):
        super().__init__()
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.N = 1_000_000
        tokens = self.N
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("baseline_triton", enable=enable_nvtx):
            baseline_elementwise(self.input, self.output)
            torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        self.input = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=100, warmup=10)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_roofline_metrics
        return compute_roofline_metrics(
            total_flops=float(getattr(self, 'total_flops', getattr(self, 'N', 1024) * 2)),
            total_bytes=float(getattr(self, 'N', 1024) * 4 * 2),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            precision="fp16",
        )

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineTritonBenchmark()


if __name__ == "__main__":
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BaselineTritonBenchmark().get_config(),
    )
    result = harness.benchmark(get_benchmark())
    print(f"Baseline (PyTorch elementwise): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
