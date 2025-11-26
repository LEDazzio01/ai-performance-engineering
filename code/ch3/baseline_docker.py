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
import os
from common.python.smoke import is_smoke_mode

from common.python.allocator_tuning import log_allocator_guidance
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


class BaselineDockerBenchmark(BaseBenchmark):
    """Simulates a non-containerized setup with blocking H2D copies."""

    def __init__(self):
        super().__init__()
        low_mem = is_smoke_mode()
        self.input_dim = 512 if low_mem else 2048
        self.hidden_dim = 1024 if low_mem else 4096
        self.output_dim = 256 if low_mem else 1024
        self.batch_size = 64 if low_mem else 256
        self.num_batches = 2 if low_mem else 4
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.host_batches: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.batch_idx = 0

    def setup(self) -> None:
        torch.manual_seed(101)
        log_allocator_guidance("ch3/baseline_docker", optimized=False)
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        ).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2)

        for _ in range(self.num_batches):
            self.host_batches.append(torch.randn(self.batch_size, self.input_dim, dtype=torch.float32))
            self.targets.append(torch.randn(self.batch_size, self.output_dim, dtype=torch.float32))
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
            x = self.to_device(host_x)  # blocking copy (tensor not pinned)
            y = self.to_device(host_y)
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
        low_mem = is_smoke_mode()
        return BenchmarkConfig(iterations=5 if low_mem else 20, warmup=1 if low_mem else 4)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_system_config_metrics
        return compute_system_config_metrics(
            numa_nodes=getattr(self, 'numa_nodes', 1),
            cpu_cores=getattr(self, 'cpu_cores', 64),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineDockerBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nBaseline Docker latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
