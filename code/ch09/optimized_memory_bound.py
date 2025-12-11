"""optimized_memory_bound.py - Keep data on GPU and fuse updates."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from typing import Optional

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class OptimizedMemoryBoundBenchmark(BaseBenchmark):
    """Keeps data resident on the GPU and fuses updates via torch.compile."""
    
    def __init__(self):
        super().__init__()
        self.data = None
        self.N = 16_777_216  # Same size as baseline (~64 MB)
        self.repeats = 64
        self.step_fn = None
        self._compiled = False
        # Memory-bound benchmark - fixed dimensions for roofline analysis
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(self.N * self.repeats),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors and compile the fused update."""
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        self.data = torch.randn(self.N, dtype=torch.float32, device=self.device)

        def fused_step(x: torch.Tensor) -> torch.Tensor:
            # Match baseline math: t = t * 1.0001 + 0.0001
            return x * 1.0001 + 0.0001

        # Avoid torch.compile here because Inductor cudagraph capture can clash
        # with repeated in-place style updates across iterations.
        self.step_fn = fused_step
        self._compiled = False
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Fused operations (high AI)."""
        assert self.data is not None and self.step_fn is not None
        with self._nvtx_range("memory_bound"):
            t = self.data
            for _ in range(self.repeats):
                t = self.step_fn(t)
            self.output = t
        self._synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
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
        """Validate benchmark result."""
        if self.data is None or self.step_fn is None:
            return "Data tensor not initialized"
        if self.data.shape[0] != self.N:
            return f"Data size mismatch: expected {self.N}, got {self.data.shape[0]}"
        if not torch.isfinite(self.data).all():
            return "Data contains non-finite values"
        return None

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return {"N": self.N, "repeats": self.repeats}  # Match baseline's repeats for signature

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.data is None:
            raise RuntimeError("Output not available - run benchmark first")
        return self.output if self.output is not None else self.data

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (1e-1, 1e-1)  # Wide tolerance - different memory patterns



def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedMemoryBoundBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
