"""Pipelined execution that overlaps microbatches across stages using CUDA streams."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import Benchmark, BenchmarkConfig
from ch18.workload_config import WORKLOAD, is_smoke_test

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")


class OptimizedPipelineParallelismBenchmark(Benchmark):
    """Optimized pipeline that uses stage-local streams for overlap."""

    def __init__(self):
        self.device = resolve_device()
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()

        self.hidden_dim = self.workload.attention_hidden_dim
        self.batch_size = self.workload.attention_batch_size
        self.sequence_length = self.workload.seq_len(self.smoke_test)
        self.micro_batches = self.workload.pipeline_micro_batches_for_mode(self.smoke_test)
        self.pipeline_depth = self.workload.pipeline_stages

        self.stages: Optional[nn.ModuleList] = None
        self.micro_inputs: list[torch.Tensor] = []
        self.stage_streams: list[torch.cuda.Stream] = []

    def setup(self) -> None:
        torch.manual_seed(42)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                ).to(self.device).half().eval()
                for _ in range(self.pipeline_depth)
            ]
        )
        self.micro_inputs = [
            torch.randn(
                self.batch_size,
                self.sequence_length,
                self.hidden_dim,
                dtype=torch.float16,
                device=self.device,
            )
            for _ in range(self.micro_batches)
        ]
        self.stage_streams = [torch.cuda.Stream() for _ in range(self.pipeline_depth)]
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        assert self.stages is not None

        with nvtx_range("optimized_pipeline_parallelism", enable=enable_nvtx):
            pending_events: list[Optional[torch.cuda.Event]] = [None] * self.pipeline_depth
            for micro in self.micro_inputs:
                activations = micro
                for stage_idx, stage in enumerate(self.stages):
                    stream = self.stage_streams[stage_idx]
                    with torch.cuda.stream(stream):
                        if stage_idx > 0 and pending_events[stage_idx - 1] is not None:
                            pending_events[stage_idx - 1].wait(stream)
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            activations = stage(activations)
                    event = torch.cuda.Event()
                    event.record(stream)
                    pending_events[stage_idx] = event
            for stream in self.stage_streams:
                stream.synchronize()

    def teardown(self) -> None:
        self.stages = None
        self.micro_inputs = []
        self.stage_streams = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=3,
            warmup=1,
            enable_memory_tracking=False,
            measurement_timeout_seconds=120,
        )

    def validate_result(self) -> Optional[str]:
        if self.stages is None:
            return "Pipeline stages missing"
        if not self.micro_inputs:
            return "Microbatch inputs missing"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedPipelineParallelismBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=3, warmup=1))
    result = harness.benchmark(OptimizedPipelineParallelismBenchmark())
    print(f"Optimized pipeline mean: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
