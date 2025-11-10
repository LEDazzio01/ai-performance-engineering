"""Shared-memory style implementation that stages tiles once."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from common.python.benchmark_harness import Benchmark, BenchmarkConfig
from ch18.workload_config import WORKLOAD, is_smoke_test

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")


class OptimizedSharedMemoryBenchmark(Benchmark):
    """Uses unfold (SIMD-friendly) to reuse data similar to shared memory tiling."""

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
        self.kernel_weights: Optional[torch.Tensor] = None

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
        self.kernel_weights = torch.randn(
            self.feature_maps,
            self.kernel,
            self.kernel,
            dtype=torch.float16,
            device=self.device,
        )
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        assert self.input is not None
        assert self.kernel_weights is not None

        with nvtx_range("optimized_shared_memory", enable=enable_nvtx):
            patches = F.unfold(
                self.input,
                kernel_size=self.kernel,
                padding=self.radius,
            ).view(
                self.batch,
                self.feature_maps,
                self.kernel * self.kernel,
                -1,
            )
            kernel_flat = self.kernel_weights.view(self.feature_maps, self.kernel * self.kernel)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                fused = torch.einsum("bckn,ck->bcn", patches, kernel_flat)
            result = fused.view(self.batch, self.feature_maps, self.spatial, self.spatial)
            _ = result.sum()

    def teardown(self) -> None:
        self.input = None
        self.kernel_weights = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=4,
            warmup=1,
            enable_memory_tracking=False,
            measurement_timeout_seconds=90,
        )

    def validate_result(self) -> Optional[str]:
        if self.input is None or self.kernel_weights is None:
            return "Inputs not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedSharedMemoryBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=4, warmup=1))
    result = harness.benchmark(OptimizedSharedMemoryBenchmark())
    print(f"Optimized shared memory mean: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
