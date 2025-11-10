"""baseline_pipeline_sequential.py - Sequential pipeline baseline (baseline).

Sequential execution of pipeline stages without overlap.
Each stage waits for the previous to complete.

Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn


from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")


class SimpleStage(nn.Module):
    """Heavier pipeline stage to highlight overlap benefits."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ffn(x)
        return self.norm(out + x)


class BaselinePipelineSequentialBenchmark(Benchmark):
    """Sequential pipeline - no overlap."""
    
    def __init__(self):
        self.device = resolve_device()
        self.stages = None
        self.inputs = None
        self.batch_size = 256
        self.hidden_dim = 1024
        self.num_stages = 4
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        """Describe workload units processed per iteration."""
        return WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size),
            samples_per_iteration=float(self.batch_size),
        )
    
    def setup(self) -> None:
        """Setup: Initialize pipeline stages."""
        torch.manual_seed(42)
        
        # Sequential pipeline stages
        self.stages = nn.ModuleList([
            SimpleStage(self.hidden_dim).to(self.device).half()
            for _ in range(self.num_stages)
        ]).eval()
        
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        
        # Warmup
        x = self.inputs
        for stage in self.stages:
            x = stage(x)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - sequential pipeline."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_pipeline_sequential", enable=enable_nvtx):
            # Sequential execution: each stage waits for previous
            x = self.inputs
            for stage in self.stages:
                x = stage(x)  # Wait for completion before next stage
                torch.cuda.synchronize()
                # Naive pipeline: copy activations back to host between stages at FP32 precision
                host_buffer = x.detach().float().to("cpu", non_blocking=False)
                torch.cuda.synchronize()
                x = host_buffer.to(self.device, non_blocking=False).half()
                torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.stages, self.inputs
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.stages is None:
            return "Stages not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselinePipelineSequentialBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Pipeline Sequential: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
