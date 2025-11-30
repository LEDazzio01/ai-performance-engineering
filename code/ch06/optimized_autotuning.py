"""optimized_autotuning.py - Optimized with autotuning."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# Import arch_config to apply Triton patch for sm_12x support
try:
    import arch_config  # noqa: F401
except ImportError:
    pass  # Continue if arch_config not available

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedAutotuningBenchmark(BaseBenchmark):
    """Optimized: uses autotuning to find optimal parameters."""
    
    def __init__(self):
        super().__init__()
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.N = 4_000_000
        self.candidates = [1024, 2048, 4096, 8192]
        self.optimal_chunk: Optional[int] = None
        self.timer_results: List[Tuple[int, float]] = []
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors and perform autotuning."""
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self.optimal_chunk = self._autotune_chunk_size()
        self._synchronize()

    def _transform(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.mul(1.75)
        out = out.add(0.1)
        return F.silu(out)

    def _autotune_chunk_size(self) -> int:
        """Benchmark several staging chunk sizes using baseline timers."""
        best = None
        best_time = float("inf")
        for chunk in self.candidates:
            total_ms = 0.0
            trials = 3
            for _ in range(trials):
                start = self._record_start()
                for offset in range(0, self.N, chunk):
                    span = min(chunk, self.N - offset)
                    window = self.input[offset : offset + span]
                    transformed = self._transform(window)
                    self.output[offset : offset + span].copy_(transformed)
                self._synchronize()
                total_ms += self._record_stop(start)
            avg_ms = total_ms / trials
            self.timer_results.append((chunk, avg_ms))
            if avg_ms < best_time:
                best_time = avg_ms
                best = chunk
        assert best is not None
        return best
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with autotuned parameters."""
        assert self.input is not None and self.output is not None and self.optimal_chunk is not None
        with self._nvtx_range("optimized_autotuning"):
            chunk = self.optimal_chunk
            for offset in range(0, self.N, chunk):
                span = min(chunk, self.N - offset)
                window = self.input[offset : offset + span]
                transformed = self._transform(window)
                self.output[offset : offset + span].copy_(transformed, non_blocking=True)
            self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_kernel_fundamentals_metrics
        return compute_kernel_fundamentals_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            num_iterations=1,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedAutotuningBenchmark()
