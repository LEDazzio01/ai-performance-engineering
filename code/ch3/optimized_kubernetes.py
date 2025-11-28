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

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from common.python.compile_utils import enable_tf32


class OptimizedKubernetesBenchmark(BaseBenchmark):
    """Prefetches device batches on a side stream and runs the step in FP16."""

    def __init__(self):
        super().__init__()
        model = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
        ).to(self.device)
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is not None:
            model = compile_fn(model, mode="reduce-overhead")
        else:
            raise RuntimeError("torch.compile is required for this benchmark")
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9)
        self.device_batches: List[torch.Tensor] = []
        self.target_batches: List[torch.Tensor] = []
        self.copy_stream = torch.cuda.Stream()
        self.cur_slot = 0
        self.next_slot = 1
        elements = 2 * 512 * 1024
        # Two half batches per iteration (inputs + targets)
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(elements),
            bytes_per_iteration=float(elements * 2),
        )

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
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
            bytes_per_iteration=self._workload.bytes_per_iteration,
        )

    def benchmark_fn(self) -> None:
        torch.cuda.current_stream().wait_stream(self.copy_stream)
        inputs = self.device_batches[self.cur_slot]
        targets = self.target_batches[self.cur_slot]

        with self._nvtx_range("optimized_kubernetes"):
            with torch.autocast("cuda", dtype=torch.float16):
                out = self.model(inputs)
                loss = torch.nn.functional.mse_loss(out, targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        self.cur_slot, self.next_slot = self.next_slot, self.cur_slot
        self._prefetch_slot(self.next_slot)
        self._synchronize()

    def teardown(self) -> None:
        self.device_batches = []
        self.target_batches = []
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=30, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_system_config_metrics
        return compute_system_config_metrics(
            numa_nodes=getattr(self, 'numa_nodes', 1),
            cpu_cores=getattr(self, 'cpu_cores', 64),
        )

    def validate_result(self) -> Optional[str]:
        if not self.device_batches:
            return "Device batches not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedKubernetesBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=10),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nOptimized Kubernetes latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
