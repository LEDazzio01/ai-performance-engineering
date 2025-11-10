"""Single-node execution without any distributed fusion."""

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


class BaselineDistributedBenchmark(Benchmark):
    """Baseline that executes each data-parallel shard sequentially on one device."""

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
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        assert self.model is not None
        assert self.inputs is not None

        with nvtx_range("baseline_distributed", enable=enable_nvtx):
            for rank in range(self.ranks):
                start = rank * self.per_rank
                end = start + self.per_rank
                shard = self.inputs[start:end]
                _ = self.model(shard)
                torch.cuda.synchronize()

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
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
    return BaselineDistributedBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=5, warmup=1))
    result = harness.benchmark(BaselineDistributedBenchmark())
    print(f"Baseline distributed mean: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
