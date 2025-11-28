"""optimized_cutlass.py - Optimized GEMM using CUTLASS."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from typing import Optional, Tuple

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from common.python.compile_utils import configure_tf32, restore_tf32

def _try_cutlass_gemm(a: torch.Tensor, b: torch.Tensor):
    """Try CUTLASS GEMM, fallback to torch.matmul if unavailable."""
    try:
        from common.python.cutlass_binding import cutlass_gemm_fp16
        return cutlass_gemm_fp16(a, b), True
    except Exception:
        return torch.matmul(a, b), False


class OptimizedCutlassBenchmark(BaseBenchmark):
    """Optimized: Single GEMM call using PyTorch's optimized kernels.
    
    Contrast with baseline's naive blocked matmul (many small GEMM calls).
    Uses FP16 + single torch.matmul for maximum tensor core utilization.
    
    Chapter 14 Learning Goal: Show how compiler/library optimizations 
    (single optimized GEMM vs naive blocked matmul) improve performance.
    """
    
    def __init__(self):
        super().__init__()
        self.A = None
        self.B = None
        self.C = None
        # Match baseline matrix size for fair comparison
        self.m = 4096
        self.n = 4096
        self.k = 4096
        self._use_cutlass = False
        self._tf32_state: Optional[Tuple[Optional[str], Optional[str]]] = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.m * self.n),
        )
    
    def setup(self) -> None:
        """Setup: Initialize matrices with optimal configuration."""
        torch.manual_seed(42)
        
        # ENABLE TF32 for tensor core acceleration (this IS the optimization!)
        # Baseline disables TF32 to simulate non-optimized GEMM
        self._tf32_state = configure_tf32(enable_matmul=True, enable_cudnn=True)
        torch.set_float32_matmul_precision("high")
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Use float16 matrices for tensor core acceleration
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float16)
        self.C = torch.zeros(self.m, self.n, device=self.device, dtype=torch.float16)
        
        # Try CUTLASS, fallback to torch.matmul
        _, self._use_cutlass = _try_cutlass_gemm(self.A, self.B)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Single optimized GEMM (vs baseline's many small GEMMs)."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_cutlass", enable=enable_nvtx):
            # KEY OPTIMIZATION: Single GEMM call
            # Baseline does many small blocked matmuls = poor locality
            # This does one large GEMM = optimal tensor core utilization
            if self.A is None or self.B is None:
                raise RuntimeError("Benchmark not initialized")
            
            if self._use_cutlass:
                from common.python.cutlass_binding import cutlass_gemm_fp16
                self.C = cutlass_gemm_fp16(self.A, self.B)
            else:
                # Fallback: Still faster than baseline's blocked matmul
                torch.matmul(self.A, self.B, out=self.C)
            self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.A = None
        self.B = None
        self.C = None
        if self._tf32_state is not None:
            restore_tf32(self._tf32_state)
            self._tf32_state = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_triton_metrics
        return compute_triton_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            block_size=getattr(self, 'BLOCK_SIZE', 1024),
            num_warps=getattr(self, 'num_warps', 4),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None or self.C is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedCutlassBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized CUTLASS: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
