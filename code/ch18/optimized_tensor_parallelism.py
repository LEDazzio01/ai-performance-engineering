"""Tensor parallel implementation that fuses shards into a single GEMM."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import torch

from common.python.benchmark_harness import Benchmark, BenchmarkConfig
from ch18.workload_config import WORKLOAD, is_smoke_test

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")


class OptimizedTensorParallelismBenchmark(Benchmark):
    """Aggregates TP shards into a fused matmul and overlaps the gather."""

    def __init__(self):
        self.device = resolve_device()
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.hidden_dim = self.workload.attention_hidden_dim
        self.batch_size = self.workload.attention_batch_size
        self.sequence_length = self.workload.seq_len(self.smoke_test)
        self.shards = self.workload.tensor_parallel_shards
        self.shard_dim = self.hidden_dim // self.shards

        self.inputs: Optional[torch.Tensor] = None
        self.weight_shards: List[torch.Tensor] = []
        self.comm_stream: Optional[torch.cuda.Stream] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        self.inputs = torch.randn(
            self.batch_size,
            self.sequence_length,
            self.hidden_dim,
            dtype=torch.float16,
            device=self.device,
        )
        self.weight_shards = [
            torch.randn(
                self.hidden_dim,
                self.shard_dim,
                dtype=torch.float16,
                device=self.device,
            )
            for _ in range(self.shards)
        ]
        self.comm_stream = torch.cuda.Stream()
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        assert self.inputs is not None
        assert self.comm_stream is not None

        with nvtx_range("optimized_tensor_parallelism", enable=enable_nvtx):
            fused_weight = torch.cat(self.weight_shards, dim=-1)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = torch.matmul(self.inputs, fused_weight)

            reduced = torch.empty_like(output)
            with torch.cuda.stream(self.comm_stream):
                reduced.copy_(output)
            self.comm_stream.synchronize()
            _ = reduced.norm()

    def teardown(self) -> None:
        self.inputs = None
        self.weight_shards = []
        self.comm_stream = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=4, warmup=1, enable_memory_tracking=False)

    def validate_result(self) -> Optional[str]:
        if self.inputs is None or not self.weight_shards:
            return "Inputs not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedTensorParallelismBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=4, warmup=1))
    result = harness.benchmark(OptimizedTensorParallelismBenchmark())
    print(f"Optimized tensor parallel mean: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
