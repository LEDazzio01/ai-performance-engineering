"""Docker optimization: pinned-memory prefetch + compute/copy overlap."""

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


class Prefetcher:
    """Double-buffered prefetcher from pinned host memory to device."""

    def __init__(self, device: torch.device, host_batches: List[torch.Tensor], targets: List[torch.Tensor]):
        self.device = device
        self.host_batches = host_batches
        self.targets = targets
        self.copy_stream = torch.cuda.Stream()
        self.buffers = [
            torch.empty_like(host_batches[0], device=device, dtype=host_batches[0].dtype),
            torch.empty_like(host_batches[0], device=device, dtype=host_batches[0].dtype),
        ]
        self.target_bufs = [
            torch.empty_like(targets[0], device=device, dtype=targets[0].dtype),
            torch.empty_like(targets[0], device=device, dtype=targets[0].dtype),
        ]
        self.cur_slot = 0
        self.next_slot = 1
        self.batch_idx = 0
        self._inflight = False
        self._prefetch()

    def _prefetch(self) -> None:
        host_idx = self.batch_idx % len(self.host_batches)
        self.batch_idx += 1
        with torch.cuda.stream(self.copy_stream):
            self.buffers[self.next_slot].copy_(self.host_batches[host_idx], non_blocking=True)
            self.target_bufs[self.next_slot].copy_(self.targets[host_idx], non_blocking=True)
        self._inflight = True

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.cuda.current_stream().wait_stream(self.copy_stream)
        if self._inflight:
            self.cur_slot, self.next_slot = self.next_slot, self.cur_slot
            self._prefetch()
        return self.buffers[self.cur_slot], self.target_bufs[self.cur_slot]


class OptimizedDockerBenchmark(Benchmark):
    """Pinned memory, asynchronous copies, and fused training step."""

    def __init__(self):
        self.device = resolve_device()
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.host_batches: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.prefetcher: Optional[Prefetcher] = None

    def setup(self) -> None:
        torch.manual_seed(101)
        self.model = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.GELU(),
            nn.Linear(4096, 1024),
        ).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9)

        for _ in range(4):
            self.host_batches.append(torch.randn(256, 2048, dtype=torch.float16, pin_memory=True))
            self.targets.append(torch.randn(256, 1024, dtype=torch.float16, pin_memory=True))
        self.prefetcher = Prefetcher(self.device, self.host_batches, self.targets)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.model is not None and self.optimizer is not None and self.prefetcher is not None

        inputs, targets = self.prefetcher.next()
        with nvtx_range("optimized_docker", enable=enable_nvtx):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = self.model(inputs)
                loss = torch.nn.functional.mse_loss(out, targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    def teardown(self) -> None:
        self.model = None
        self.optimizer = None
        self.host_batches = []
        self.targets = []
        self.prefetcher = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=4)

    def validate_result(self) -> Optional[str]:
        if self.prefetcher is None:
            return "Prefetcher not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedDockerBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nOptimized Docker latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
