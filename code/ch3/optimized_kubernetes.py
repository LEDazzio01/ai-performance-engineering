"""Kubernetes optimization: overlap data provisioning with training work."""

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
from common.python.compile_utils import enable_tf32


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch3 kubernetes example")
    return torch.device("cuda")


class OptimizedKubernetesBenchmark(Benchmark):
    """Prefetches device batches on a side stream and runs the step in FP16."""

    def __init__(self):
        self.device = resolve_device()
        model = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
        ).to(self.device)
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is not None:
            try:
                model = compile_fn(model, mode="reduce-overhead")
            except Exception:
                pass
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9)
        self.device_batches: List[torch.Tensor] = []
        self.target_batches: List[torch.Tensor] = []
        self.copy_stream = torch.cuda.Stream()
        self.cur_slot = 0
        self.next_slot = 1

    def _prefetch_slot(self, slot: int) -> None:
        with torch.cuda.stream(self.copy_stream):
            self.device_batches[slot].normal_()
            self.target_batches[slot].normal_()

    def setup(self) -> None:
        torch.manual_seed(314)
        enable_tf32()
        self.device_batches = [
            torch.empty(512, 1024, device=self.device, dtype=torch.float16)
            for _ in range(2)
        ]
        self.target_batches = [
            torch.empty(512, 1024, device=self.device, dtype=torch.float16)
            for _ in range(2)
        ]
        for slot in range(2):
            self._prefetch_slot(slot)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        torch.cuda.current_stream().wait_stream(self.copy_stream)
        inputs = self.device_batches[self.cur_slot]
        targets = self.target_batches[self.cur_slot]

        with nvtx_range("optimized_kubernetes", enable=enable_nvtx):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = self.model(inputs)
                loss = torch.nn.functional.mse_loss(out, targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        self.cur_slot, self.next_slot = self.next_slot, self.cur_slot
        self._prefetch_slot(self.next_slot)

    def teardown(self) -> None:
        self.device_batches = []
        self.target_batches = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=30, warmup=5)

    def validate_result(self) -> Optional[str]:
        if not self.device_batches:
            return "Device batches not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedKubernetesBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nOptimized Kubernetes latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
