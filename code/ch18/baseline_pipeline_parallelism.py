"""Pipeline without overlap: flushes each microbatch through all stages serially."""

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


class BaselinePipelineParallelismBenchmark(Benchmark):
    """Sequential pipeline that never overlaps stage execution."""

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
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        assert self.stages is not None

        with nvtx_range("baseline_pipeline_parallelism", enable=enable_nvtx):
            for micro in self.micro_inputs:
                activations = micro
                for stage in self.stages:
                    activations = stage(activations)
                    torch.cuda.synchronize()

    def teardown(self) -> None:
        self.stages = None
        self.micro_inputs = []
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
    return BaselinePipelineParallelismBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=3, warmup=1))
    result = harness.benchmark(BaselinePipelineParallelismBenchmark())
    print(f"Baseline pipeline mean: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
