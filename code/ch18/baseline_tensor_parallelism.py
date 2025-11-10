"""Tensor parallel baseline that executes shards sequentially."""

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


class BaselineTensorParallelismBenchmark(Benchmark):
    """Processes each TP shard sequentially without any gather fusion."""

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
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        assert self.inputs is not None

        with nvtx_range("baseline_tensor_parallelism", enable=enable_nvtx):
            partial_outputs = []
            for weight in self.weight_shards:
                partial = torch.matmul(self.inputs, weight)
                partial_outputs.append(partial)
                torch.cuda.synchronize()
            output = torch.cat(partial_outputs, dim=-1)
            _ = output.norm()

    def teardown(self) -> None:
        self.inputs = None
        self.weight_shards = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=4, warmup=1, enable_memory_tracking=False)

    def validate_result(self) -> Optional[str]:
        if self.inputs is None or not self.weight_shards:
            return "Inputs not initialized"
        return None


def get_benchmark() -> Benchmark:
    return BaselineTensorParallelismBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=4, warmup=1))
    result = harness.benchmark(BaselineTensorParallelismBenchmark())
    print(f"Baseline tensor parallel mean: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
