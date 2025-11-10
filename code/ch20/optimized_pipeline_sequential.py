"""optimized_pipeline_sequential.py - Pipeline overlap optimization (optimized).

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


class OptimizedPipelineOverlapBenchmark(Benchmark):
    """Pipeline overlap - stages execute concurrently."""
    
    def __init__(self):
        self.device = resolve_device()
        self.stages = None
        self.stage_streams = None
        self.inputs = None
        self.batch_size = 256
        self.hidden_dim = 1024
        self.num_stages = 4
        self.num_micro_batches = 8
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        """Describe workload units processed per iteration."""
        return WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size),
            samples_per_iteration=float(self.batch_size),
        )
    
    def setup(self) -> None:
        """Setup: Initialize pipeline stages and streams."""
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        
        self.stages = nn.ModuleList(
            [SimpleStage(self.hidden_dim).to(self.device).half() for _ in range(self.num_stages)]
        ).eval()
        self.stage_streams = [torch.cuda.Stream() for _ in range(self.num_stages)]
        
        self.inputs = torch.randn(
            self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16
        )
        
        # Warmup each stage individually
        data = self.inputs
        with torch.no_grad():
            for stage in self.stages:
                data = stage(data)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - pipeline with overlap."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("pipeline_sequential", enable=enable_nvtx):
            micro_batches = list(self.inputs.chunk(self.num_micro_batches))
            with torch.no_grad():
                self._run_pipeline(micro_batches)
            torch.cuda.synchronize()

    def _run_pipeline(self, micro_batches: list[torch.Tensor]) -> None:
        if self.stage_streams is None or self.stages is None:
            raise RuntimeError("Pipeline stages not initialized")
        num_micro = len(micro_batches)
        ready_events = [
            [torch.cuda.Event(blocking=False) for _ in range(num_micro)]
            for _ in range(self.num_stages)
        ]
        activations = [dict() for _ in range(self.num_stages)]
        total_steps = num_micro + self.num_stages - 1
        for step in range(total_steps):
            for stage_idx in range(self.num_stages):
                micro_idx = step - stage_idx
                if micro_idx < 0 or micro_idx >= num_micro:
                    continue
                stream = self.stage_streams[stage_idx]
                with torch.cuda.stream(stream):
                    if stage_idx == 0:
                        tensor = micro_batches[micro_idx]
                    else:
                        stream.wait_event(ready_events[stage_idx - 1][micro_idx])
                        tensor = activations[stage_idx - 1].pop(micro_idx)
                    tensor = self.stages[stage_idx](tensor)
                    activations[stage_idx][micro_idx] = tensor
                    ready_events[stage_idx][micro_idx].record(stream)
        for ev in ready_events[-1]:
            ev.synchronize()
        activations[-1].clear()

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.stages, self.stage_streams, self.inputs
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
    timing = result.timing
    if timing:
        print(f"\nOptimized Pipeline Overlap: {timing.mean_ms:.3f} ms")
    else:
        print("No timing data available")
