"""Kubernetes baseline: per-iteration CPU orchestration with new tensors."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from common.python.benchmark_harness import Benchmark, BenchmarkConfig


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch3 kubernetes example")
    return torch.device("cuda")


class BaselineKubernetesBenchmark(Benchmark):
    """Allocates new tensors + launches multiple kernels every iteration."""

    def __init__(self):
        self.device = resolve_device()
        self.model = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
        ).to(self.device)

    def setup(self) -> None:
        torch.manual_seed(314)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("baseline_kubernetes", enable=enable_nvtx):
            host_data = torch.randn(512, 1024, dtype=torch.float32)
            host_target = torch.randn(512, 1024, dtype=torch.float32)
            data = host_data.to(self.device, non_blocking=False)
            target = host_target.to(self.device, non_blocking=False)
            torch.cuda.synchronize()
            out = self.model(data)
            torch.nn.functional.mse_loss(out, target).backward()
            for p in self.model.parameters():
                p.grad = None

    def teardown(self) -> None:
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=30, warmup=5)

    def validate_result(self) -> Optional[str]:
        return None


def get_benchmark() -> Benchmark:
    return BaselineKubernetesBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nBaseline Kubernetes latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
