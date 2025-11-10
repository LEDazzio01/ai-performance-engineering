"""Data-parallel execution that fuses shards and overlaps reduction."""

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


class OptimizedDistributedBenchmark(Benchmark):
    """Optimized variant that batches all ranks and performs fused reduction."""

    def __init__(self):
        self.device = resolve_device()
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()

        self.hidden_dim = self.workload.attention_hidden_dim
        self.global_batch = self.workload.distributed_global_batch
        self.ranks = self.workload.distributed_ranks
        self.per_rank = self.global_batch // self.ranks

        self.model: Optional[nn.Sequential] = None
        self.inputs: Optional[torch.Tensor] = None
        self.comm_stream: Optional[torch.cuda.Stream] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        ).to(self.device).half().eval()
        self.inputs = torch.randn(
            self.global_batch,
            self.hidden_dim,
            dtype=torch.float16,
            device=self.device,
        )
        self.comm_stream = torch.cuda.Stream()
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        assert self.model is not None
        assert self.inputs is not None
        assert self.comm_stream is not None

        with nvtx_range("optimized_distributed", enable=enable_nvtx):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.model(self.inputs)

            shards = outputs.view(self.ranks, self.per_rank, -1)
            reduced = torch.zeros_like(shards[0])

            with torch.cuda.stream(self.comm_stream):
                reduced.copy_(shards.mean(dim=0))

            self.comm_stream.synchronize()
            _ = reduced.norm()

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.comm_stream = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=5,
            warmup=1,
            enable_memory_tracking=False,
            measurement_timeout_seconds=90,
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.inputs is None:
            return "Input tensor missing"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedDistributedBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=5, warmup=1))
    result = harness.benchmark(OptimizedDistributedBenchmark())
    print(f"Optimized distributed mean: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
