"""Docker baseline: host batches copied synchronously each iteration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from common.python.benchmark_harness import Benchmark, BenchmarkConfig


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch3 docker example")
    return torch.device("cuda")


class BaselineDockerBenchmark(Benchmark):
    """Simulates a non-containerized setup with blocking H2D copies."""

    def __init__(self):
        self.device = resolve_device()
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.host_batches: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.batch_idx = 0

    def setup(self) -> None:
        torch.manual_seed(101)
        self.model = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
        ).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2)

        for _ in range(4):
            self.host_batches.append(torch.randn(256, 2048, dtype=torch.float32))
            self.targets.append(torch.randn(256, 1024, dtype=torch.float32))
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.model is not None and self.optimizer is not None

        idx = self.batch_idx % len(self.host_batches)
        host_x = self.host_batches[idx]
        host_y = self.targets[idx]
        self.batch_idx += 1

        with nvtx_range("baseline_docker", enable=enable_nvtx):
            x = host_x.to(self.device)  # blocking copy
            y = host_y.to(self.device)
            out = self.model(x)
            loss = torch.nn.functional.mse_loss(out, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

    def teardown(self) -> None:
        self.model = None
        self.optimizer = None
        self.host_batches = []
        self.targets = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=4)

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    return BaselineDockerBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nBaseline Docker latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
