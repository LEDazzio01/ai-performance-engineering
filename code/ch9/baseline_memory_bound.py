"""baseline_memory_bound.py - Memory-bound kernel (low arithmetic intensity)."""

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


class BaselineMemoryBoundBenchmark(BaseBenchmark):
    """Simple element-wise ops with low arithmetic intensity."""

    def __init__(self):
        super().__init__()
        self.tensor: Optional[torch.Tensor] = None
        self.repeats = 64
        self.N = 16_777_216  # ~64 MB
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(self.N * self.repeats),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        self.tensor = torch.randn(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("baseline_memory_bound", enable=enable_nvtx):
            t = self.tensor
            for _ in range(self.repeats):
                t = t * 1.0001 + 0.0001
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
        from common.python.benchmark_metrics import compute_roofline_metrics
        return compute_roofline_metrics(
            total_flops=float(getattr(self, 'total_flops', getattr(self, 'N', 1024) * 2)),
            total_bytes=float(getattr(self, 'N', 1024) * 4 * 2),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            precision="fp16",
        )
}

    def validate_result(self) -> Optional[str]:
        if self.tensor is None:
            return "Tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineMemoryBoundBenchmark()


if __name__ == "__main__":
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BaselineMemoryBoundBenchmark().get_config(),
    )
    result = harness.benchmark(get_benchmark())
    print(f"Baseline memory-bound: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
