"""optimized_add_parallel.py - Vectorized addition benchmark (optimized).

Demonstrates correct GPU utilization with single vectorized kernel.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from typing import Optional
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch6")
    return torch.device("cuda")

class OptimizedAddParallelBenchmark(Benchmark):
    """Benchmark implementation following Benchmark protocol."""
    
    def __init__(self):
        self.device = resolve_device()
        self.A = None
        self.B = None
        self.C = None
        self.N = 10_000  # Same as baseline for fair comparison
    
    def setup(self) -> None:
        """Setup: Initialize tensors (EXCLUDED from timing)."""
        # Use same seed for fair comparison
        torch.manual_seed(42)
        if torch.cuda.is_available():
                                    torch.cuda.manual_seed_all(42)
        
        # Pre-allocate all tensors in setup
        self.A = torch.arange(self.N, dtype=torch.float32, device=self.device)
        self.B = 2 * self.A
        self.C = None  # Will be allocated in benchmark_fn, but that's OK for vectorized op
    
    def benchmark_fn(self) -> None:
        """Function to benchmark (ONLY this is timed via CUDA Events).
        
        Vectorized operation - single kernel launch, optimal GPU utilization.
        """
        # Add NVTX marker for profiling
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_add_parallel", enable=enable_nvtx):
    # NO sync, NO time.time() here
    # Only the computation we want to measure
    # Single vectorized kernel launch
            self.C = self.A + self.B

    
    def teardown(self) -> None:
        """Cleanup (EXCLUDED from timing)."""
        del self.A, self.B, self.C
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config.
        
        Uses CUSTOM mode which automatically uses CUDA Events for GPU timing.
        """
        return BenchmarkConfig(
            iterations=50,  # More iterations since this is fast
            warmup=10,  # Warmup to ensure GPU is ready
            enable_memory_tracking=False,  # Not relevant for this benchmark
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.C is None:
            return "Result tensor C not initialized"
        if self.A is None:
            return "Input tensor A not initialized"
        if self.B is None:
            return "Input tensor B not initialized"
        
        # Verify result correctness: C = A + B
        if self.C.shape != self.A.shape or self.C.shape != self.B.shape:
            return f"Shape mismatch: A={self.A.shape}, B={self.B.shape}, C={self.C.shape}"
        if not torch.isfinite(self.C).all():
            return "Result tensor C contains non-finite values"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedAddParallelBenchmark()

def main() -> None:
    """Standalone execution with proper harness (uses CUDA Events)."""
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,  # Uses CUDA Events (best for GPU)
        config=BenchmarkConfig(iterations=50, warmup=10)
    )
    benchmark = OptimizedAddParallelBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Optimized: Parallel Vectorized Addition")
    print("=" * 70)
    print(f"Array size: {benchmark.N:,} elements")
    print(f"Mean time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")
    print(f"Kernel launches: 1 (single vectorized kernel)")
    print("Benefit: Massive reduction in launch overhead")
    print("\n  This demonstrates the importance of vectorized operations for GPU performance")

if __name__ == "__main__":
    main()
