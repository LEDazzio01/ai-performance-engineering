"""Optimized distributed example with overlapped copies and compute."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import Benchmark, BenchmarkConfig


class OptimizedDistributedBenchmark(Benchmark):
    """Uses CUDA streams to overlap host copies with compute, mimicking NCCL pipelining."""

    virtual_ranks = 4
    batch_per_rank = 2048
    feature_dim = 1024
    nvtx_label = "optimized_distributed"

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for Chapter 8 distributed example")
        self.device = torch.device("cuda")
        self.model: Optional[nn.Module] = None
        self.host_inputs: List[torch.Tensor] = []
        self.device_buffers: List[torch.Tensor] = []
        self.partial_outputs: List[torch.Tensor] = []
        self.streams: List[torch.cuda.Stream] = []

    def setup(self) -> None:
        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Linear(self.feature_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, self.feature_dim),
        ).to(self.device).eval()

        self.host_inputs = [
            torch.randn(self.batch_per_rank, self.feature_dim, dtype=torch.float32).pin_memory()
            for _ in range(self.virtual_ranks)
        ]
        self.device_buffers = [
            torch.empty_like(self.host_inputs[0], device=self.device)
            for _ in range(self.virtual_ranks)
        ]
        self.partial_outputs = [
            torch.empty_like(self.host_inputs[0], device=self.device)
            for _ in range(self.virtual_ranks)
        ]
        self.streams = [torch.cuda.Stream(device=self.device) for _ in range(self.virtual_ranks)]

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range(self.nvtx_label, enable=enable_nvtx):
            torch.cuda.synchronize()
            with torch.no_grad():
                for rank in range(self.virtual_ranks):
                    stream = self.streams[rank]
                    with torch.cuda.stream(stream):
                        self.device_buffers[rank].copy_(self.host_inputs[rank], non_blocking=True)
                        self.partial_outputs[rank].copy_(self.model(self.device_buffers[rank]))
            torch.cuda.synchronize()

    def teardown(self) -> None:
        self.model = None
        self.host_inputs = []
        self.device_buffers = []
        self.partial_outputs = []
        self.streams = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedDistributedBenchmark()


def main() -> None:
    from common.python.benchmark_harness import BenchmarkConfig, BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5),
    )
    benchmark = OptimizedDistributedBenchmark()
    result = harness.benchmark(benchmark)
    print("=" * 70)
    print("Optimized Distributed")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()

