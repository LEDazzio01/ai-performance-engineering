"""Baseline that repeatedly reloads data from global memory."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

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


class BaselineSharedMemoryBenchmark(Benchmark):
    """Sliding-window accumulation implemented with repeated global reads."""

    def __init__(self):
        self.device = resolve_device()
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.batch = self.workload.attention_batch_size
        self.feature_maps = self.workload.shared_feature_maps
        self.spatial = self.workload.shared_spatial
        self.kernel = self.workload.shared_kernel_size
        self.radius = self.kernel // 2
        self.input: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        self.input = torch.randn(
            self.batch,
            self.feature_maps,
            self.spatial,
            self.spatial,
            dtype=torch.float16,
            device=self.device,
        )
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        assert self.input is not None

        with nvtx_range("baseline_shared_memory", enable=enable_nvtx):
            acc = torch.zeros_like(self.input)
            for dy in range(-self.radius, self.radius + 1):
                for dx in range(-self.radius, self.radius + 1):
                    shifted = torch.roll(self.input, shifts=(dy, dx), dims=(2, 3)).contiguous()
                    acc.add_(shifted)
                    torch.cuda.synchronize()
            if torch.isnan(acc).any():
                raise RuntimeError("Unexpected NaN")

    def teardown(self) -> None:
        self.input = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=4,
            warmup=1,
            enable_memory_tracking=False,
            measurement_timeout_seconds=90,
        )

    def validate_result(self) -> Optional[str]:
        if self.input is None:
            return "Input tensor missing"
        return None


def get_benchmark() -> Benchmark:
    return BaselineSharedMemoryBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=4, warmup=1))
    result = harness.benchmark(BaselineSharedMemoryBenchmark())
    print(f"Baseline shared memory mean: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
