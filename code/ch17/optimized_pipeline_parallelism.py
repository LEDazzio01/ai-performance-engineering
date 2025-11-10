"""optimized_pipeline_parallelism.py - Optimized pipeline parallelism across GPUs.

Demonstrates pipeline parallelism by splitting model layers across multiple GPUs.
Pipeline parallelism: Splits model layers across GPUs for parallel processing of microbatches.
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

from typing import Optional, List

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch17")
    return torch.device("cuda")


class OptimizedPipelineParallelismBenchmark(Benchmark):
    """Optimized: Pipeline parallelism with layers split across GPUs."""

    def __init__(self):
        self.device = resolve_device()
        self.pipeline_stages: List[nn.Module] = []
        self.hidden_size = 1024
        self.batch_size = 256
        self.micro_batches = 4
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.stage_streams: List[torch.cuda.Stream] = []
        self.stage_events: List[List[torch.cuda.Event]] = []
        self.microbatch_inputs: Optional[List[torch.Tensor]] = None

    def setup(self) -> None:
        """Setup: Initialize pipeline stages across multiple GPUs."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()

        torch.manual_seed(42)
        layers_per_stage = [
            [nn.Linear(self.hidden_size, self.hidden_size * 4), nn.GELU()],
            [nn.Linear(self.hidden_size * 4, self.hidden_size * 4), nn.GELU()],
            [nn.Linear(self.hidden_size * 4, self.hidden_size * 2), nn.GELU()],
            [nn.Linear(self.hidden_size * 2, self.hidden_size)],
        ]

        self.pipeline_stages = []
        for stage_id, layer_stack in enumerate(layers_per_stage):
            gpu_id = stage_id % self.num_gpus
            stage = nn.Sequential(*layer_stack).to(torch.device(f"cuda:{gpu_id}"), dtype=torch.bfloat16).eval()
            self.pipeline_stages.append(stage)

        self.microbatch_inputs = torch.randn(
            self.batch_size, self.hidden_size, device=torch.device("cuda:0"), dtype=torch.bfloat16
        ).chunk(self.micro_batches, dim=0)

        self.stage_streams = [torch.cuda.Stream(priority=-1) for _ in self.pipeline_stages]
        self.stage_events = [
            [torch.cuda.Event(enable_timing=False) for _ in range(self.micro_batches)]
            for _ in self.pipeline_stages
        ]
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Pipeline parallelism processing across multiple GPUs."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        if not self.pipeline_stages or self.microbatch_inputs is None:
            raise RuntimeError("Pipeline not initialized")

        num_stages = len(self.pipeline_stages)
        stage_buffers: List[List[Optional[torch.Tensor]]] = [
            [None for _ in range(self.micro_batches)] for _ in range(num_stages + 1)
        ]
        stage_buffers[0] = list(self.microbatch_inputs)

        stage_devices = [next(stage.parameters()).device for stage in self.pipeline_stages]

        with nvtx_range("optimized_pipeline_parallelism", enable=enable_nvtx):
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for micro_idx in range(self.micro_batches + num_stages - 1):
                    for stage_idx, stage in enumerate(self.pipeline_stages):
                        chunk_idx = micro_idx - stage_idx
                        if chunk_idx < 0 or chunk_idx >= self.micro_batches:
                            continue
                        stream = self.stage_streams[stage_idx]
                        with torch.cuda.stream(stream):
                            if stage_idx > 0:
                                stream.wait_event(self.stage_events[stage_idx - 1][chunk_idx])
                            x = stage_buffers[stage_idx][chunk_idx]
                            if x is None:
                                continue
                            out = stage(x.to(stage_devices[stage_idx]))
                            next_stage_idx = stage_idx + 1
                            if next_stage_idx < len(stage_devices):
                                next_device = stage_devices[next_stage_idx]
                                if next_device != stage_devices[stage_idx]:
                                    out = out.to(next_device)
                            stage_buffers[next_stage_idx][chunk_idx] = out
                            self.stage_events[stage_idx][chunk_idx].record(stream)

        for stream in self.stage_streams:
            stream.synchronize()

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.pipeline_stages = []
        self.microbatch_inputs = None
        self.stage_streams = []
        self.stage_events = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=12,
            warmup=2,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if not self.pipeline_stages:
            return "Pipeline stages not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedPipelineParallelismBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
