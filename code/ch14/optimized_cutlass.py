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

from common.python.cutlass_binding import cutlass_gemm_fp16


class OptimizedCutlassBenchmark(BaseBenchmark):
    """Optimized: GEMM using CUTLASS library.
    
    CUTLASS: Uses CUTLASS backend for hardware-optimized GEMM kernels.
    Leverages tensor cores and optimized memory access patterns for better performance.
    """
    
    def __init__(self):
        super().__init__()
        self.A = None
        self.B = None
        # Match baseline matrix size for fair comparison
        self.m = 4096
        self.n = 4096
        self.k = 4096
        self._tf32_state: Optional[Tuple[Optional[str], Optional[str]]] = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.m * self.n),
        )
    
    def setup(self) -> None:
        """Setup: Initialize matrices with optimal configuration for CUTLASS."""
        torch.manual_seed(42)
        
        # Match baseline TF32 settings for fair comparison
        # Disable TF32 to isolate CUTLASS optimization effect (not TF32 vs non-TF32)
        self._tf32_state = configure_tf32(enable_matmul=False, enable_cudnn=False)
        torch.set_float32_matmul_precision("high")
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Use float16 matrices for CUTLASS GEMM (matches baseline for fair comparison)
        # CUTLASS is optimized for FP16/Tensor Core acceleration
        # Same TF32 settings as baseline to isolate CUTLASS library effect
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float16)
        
        # Warmup the CUTLASS kernel to ensure kernels are cached before measurement
        try:
            _ = cutlass_gemm_fp16(self.A, self.B)
            torch.cuda.synchronize()
        except Exception as exc:
            raise RuntimeError(
                "SKIPPED: CUTLASS GEMM extension missing "
                "(install nvidia-cutlass-dsl>=4.2 to enable this benchmark)"
            ) from exc
    
    def benchmark_fn(self) -> None:
        """Benchmark: CUTLASS-optimized GEMM."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_cutlass", enable=enable_nvtx):
            # Optimization: CUTLASS-optimized GEMM kernel
            # CUTLASS provides hardware-optimized kernels leveraging Tensor Cores
            # Same FP16 precision and TF32 settings as baseline for fair comparison
            # This isolates the CUTLASS library optimization effect
            if self.A is None or self.B is None:
                raise RuntimeError("Benchmark not initialized")
            _ = cutlass_gemm_fp16(self.A, self.B)
            self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.A = None
        self.B = None
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
}

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None:
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
