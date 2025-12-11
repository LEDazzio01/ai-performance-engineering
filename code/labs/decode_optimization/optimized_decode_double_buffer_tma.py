"""Non-Triton warp-specialized/TMA-backed variant using the moe_cuda optimized kernel."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.harness.cuda_capabilities import tma_support_status
from labs.moe_cuda.optimized_decode_kernel import OptimizedDecodeKernelBenchmark  # noqa: E402
from labs.moe_cuda.decode_kernels import is_optimized_available, get_optimized_error


class NanoChatWarpSpecializedCudaBenchmark(OptimizedDecodeKernelBenchmark):
    """Reuse the moe_cuda TMA double-buffered kernel as a non-Triton WS/TMA path."""

    def __init__(self) -> None:
        super().__init__()
        # TMA kernel constraints: TILE_N=32, CHUNK_M=32
        # cols must be compatible with TMA tensor map encoding (divisible by tile size)
        # Using cols=1024 which works with the TMA constraints
        self.rows = 4096
        self.cols = 1024  # Keep same as base class - larger sizes may fail TMA encoding

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()



def get_benchmark() -> BaseBenchmark:
    """Return the NanoChat warp-specialized TMA benchmark.
    
    TMA is required on Blackwell B200 - no fallbacks.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for TMA decode kernel")
    
    supported, reason = tma_support_status()
    if not supported:
        raise RuntimeError(f"TMA decode kernel unavailable: {reason}")
    
    if not is_optimized_available():
        error = get_optimized_error() or "Unknown error"
        raise RuntimeError(f"TMA optimized kernel not available: {error}")
    
    return NanoChatWarpSpecializedCudaBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
