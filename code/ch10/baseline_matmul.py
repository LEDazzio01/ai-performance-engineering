"""baseline_matmul.py - Baseline FP32 matmul with serial tiling.

Demonstrates an intentionally under-optimized GEMM that processes tiles
sequentially with FP32 accumulate. Highlights the cost of redundant reads,
lack of tensor cores, and absence of CUDA Graph capture.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

try:
    import ch10.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")


class BaselineMatmulBenchmark(Benchmark):
    """Baseline: FP32 matmul with serialized tiling and no tensor cores."""

    def __init__(self):
        self.device = resolve_device()
        self.A: torch.Tensor | None = None
        self.B: torch.Tensor | None = None
        self.C: torch.Tensor | None = None
        self.n = 8192
        self.tile_k = 128

    def setup(self) -> None:
        """Setup: initialize FP32 matrices and scratch buffer."""
        torch.manual_seed(42)
        self.A = torch.randn(self.n, self.n, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.n, self.n, device=self.device, dtype=torch.float32)
        self.C = torch.zeros(self.n, self.n, device=self.device, dtype=torch.float32)
        self._chunked_matmul()
        torch.cuda.synchronize()

    def _chunked_matmul(self) -> None:
        """Multiply using many FP32 tiles to emphasize poor reuse."""
        assert self.A is not None and self.B is not None and self.C is not None
        self.C.zero_()
        for k in range(0, self.n, self.tile_k):
            k_end = min(k + self.tile_k, self.n)
            a_tile = self.A[:, k:k_end]
            b_tile = self.B[k:k_end, :]
            # Repeated addmm launches mimic per-stage kernel launches.
            self.C.addmm_(a_tile, b_tile, beta=1.0, alpha=1.0)

    def benchmark_fn(self) -> None:
        """Benchmark: serialized FP32 matmul tiles."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("matmul", enable=enable_nvtx):
            if self.A is None or self.B is None or self.C is None:
                raise RuntimeError("Matrices not initialized")
            self._chunked_matmul()
            torch.cuda.synchronize(self.device)


    def teardown(self) -> None:
        """Teardown: clean up tensors."""
        self.A = None
        self.B = None
        self.C = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None or self.C is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineMatmulBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Matmul (Tiled FP32): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Note: Serialized addmm tiles emulate poor scheduling and FP32 only math.")
