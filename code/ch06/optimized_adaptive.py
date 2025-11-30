"""optimized_adaptive.py - Optimized with adaptive runtime optimization."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedAdaptiveBenchmark(BaseBenchmark):
    """Optimized: Adaptive runtime optimization."""
    
    def __init__(self):
        super().__init__()
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.N = 4_000_000
        self.adaptive_chunk: Optional[int] = None
        self.prefetch_stream: Optional[torch.cuda.Stream] = None
        self.stage_buffers: list[torch.Tensor] = []
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N),
        )
    
    def setup(self) -> None:
        """Setup: Initialize with adaptive configuration."""
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        props = torch.cuda.get_device_properties(self.device)
        sm_count = props.multi_processor_count
        warp_allocation = props.warp_size * sm_count
        # Choose a chunk that keeps at least two CTAs per SM resident but still fits L2.
        self.adaptive_chunk = min(self.N, warp_allocation * 256)
        self.prefetch_stream = torch.cuda.Stream()
        self.stage_buffers = [
            torch.empty(self.adaptive_chunk, device=self.device, dtype=torch.float32)
            for _ in range(2)
        ]
        self._synchronize()

    def _transform(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.mul(1.75)
        out = out.add(0.1)
        return F.silu(out)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Adaptive optimization operations."""
        assert self.prefetch_stream is not None
        assert self.input is not None and self.output is not None
        assert self.adaptive_chunk is not None
        with self._nvtx_range("optimized_adaptive"):
            chunk_plan = []
            start = 0
            while start < self.N:
                span = min(self.adaptive_chunk, self.N - start)
                chunk_plan.append((start, span))
                start += span
            for idx, (start, span) in enumerate(chunk_plan):
                buf = self.stage_buffers[idx % len(self.stage_buffers)]
                slice_buf = buf[:span]
                next_slice = self.input[start : start + span]
                with torch.cuda.stream(self.prefetch_stream):
                    slice_buf.copy_(next_slice, non_blocking=True)
                torch.cuda.current_stream().wait_stream(self.prefetch_stream)
                transformed = self._transform(slice_buf)
                self.output[start : start + span].copy_(transformed, non_blocking=True)
            self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        self.stage_buffers = []
        self.prefetch_stream = None
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
    return OptimizedAdaptiveBenchmark()
