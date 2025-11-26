"""baseline_matmul_pytorch.py - PyTorch matmul baseline (baseline).

Standard PyTorch matrix multiplication without CUTLASS optimization.
Good baseline but can be optimized with specialized GEMM kernels.

Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch


from typing import Optional

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")


class BaselineMatmulPyTorchBenchmark(BaseBenchmark):
    """PyTorch matmul baseline - standard GEMM."""
    
    def __init__(self):
        super().__init__()
        self.A = None
        self.B = None
        self.C = None
        self.bias = None
        self.m = 2048
        self.n = 2048
        self.k = 2048
        tokens = self.m * self.n
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize matrices."""
        torch.manual_seed(42)
        
        # Standard PyTorch matmul
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float32)
        self.C = torch.empty(self.m, self.n, device=self.device, dtype=torch.float32)
        self.bias = torch.randn(self.m, self.n, device=self.device, dtype=torch.float32)
        
        # Warmup
        _ = torch.relu(torch.matmul(self.A, self.B) + self.bias)
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - PyTorch matmul."""
        assert self.A is not None and self.B is not None and self.bias is not None
        with self._nvtx_range("baseline_matmul_pytorch"):
            # Standard PyTorch matrix multiplication
            out = torch.matmul(self.A, self.B)
            self.C = torch.relu(out + self.bias)
        self._synchronize()

    def teardown(self) -> None:
        """Cleanup."""
        del self.A, self.B, self.C, self.bias
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=180,  # torch.compile compilation can take 60-120 seconds
        )
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None or self.bias is None:
            return "Tensors not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineMatmulPyTorchBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline PyTorch Matmul: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
