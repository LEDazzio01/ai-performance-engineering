"""optimized_pipeline_overlap.py - Pipeline overlap optimization (optimized).

Pipeline stages execute with overlap using CUDA streams.
Hides latency by overlapping computation stages.

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
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")


class SimpleStage(nn.Module):
    """Simple pipeline stage."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(x))


class OptimizedPipelineOverlapBenchmark(Benchmark):
    """Pipeline overlap - stages execute concurrently."""
    
    def __init__(self):
        self.device = resolve_device()
        self.stages = None
        self.inputs = None
        self.streams = None
        self.batch_size = 8
        self.hidden_dim = 1024
        self.num_stages = 4
    
    def setup(self) -> None:
        """Setup: Initialize pipeline stages and streams."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        
        # Pipeline stages
        self.stages = nn.ModuleList([
            SimpleStage(self.hidden_dim).to(self.device).half()
            for _ in range(self.num_stages)
        ]).eval()
        
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        
        # Create streams for each stage to enable overlap
        self.streams = [torch.cuda.Stream() for _ in range(self.num_stages)]
        
        # Warmup
        x = self.inputs
        for stage in self.stages:
            x = stage(x)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - pipeline with overlap."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_pipeline_overlap", enable=enable_nvtx):
            # Pipeline overlap: stages execute concurrently
            # Split batch across stages for overlap
            batch_per_stage = self.batch_size // self.num_stages
            
            # Launch stages on different streams
            outputs = []
            for i, stage in enumerate(self.stages):
                with torch.cuda.stream(self.streams[i]):
                    # Process subset of batch on this stream
                    start_idx = i * batch_per_stage
                    end_idx = start_idx + batch_per_stage if i < self.num_stages - 1 else self.batch_size
                    stage_input = self.inputs[start_idx:end_idx]
                    stage_output = stage(stage_input)
                    outputs.append(stage_output)
            
            # Synchronize all streams
            for stream in self.streams:
                stream.synchronize()

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.stages, self.inputs, self.streams
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
    return OptimizedPipelineOverlapBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Pipeline Overlap: {result.mean_ms:.3f} ms")

