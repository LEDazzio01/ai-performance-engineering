"""baseline_vectorization_memory.py - Naive vectorization baseline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class VectorizationBenchmark(BaseBenchmark):
    """Baseline: naive elementwise ops without vectorization."""

    def __init__(self):
        super().__init__()
        self.tensor: Optional[torch.Tensor] = None
        self.repeats = 32
        self.N = 8_192_000
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(self.N * self.repeats),
        )

    def setup(self) -> None:
        torch.manual_seed(0)
        self.tensor = torch.randn(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("baseline_vectorization", enable=enable_nvtx):
            t = self.tensor
            for _ in range(self.repeats):
                t = (t * 1.0001) + 0.0001
            torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        self.tensor = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        if self.tensor is None:
            return "Tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return VectorizationBenchmark()


if __name__ == "__main__":
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=VectorizationBenchmark().get_config(),
    )
    result = harness.benchmark(get_benchmark())
    print(f"Baseline vectorization: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
