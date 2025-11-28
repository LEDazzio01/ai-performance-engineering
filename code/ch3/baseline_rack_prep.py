"""Rack baseline: pageable staging buffers and topology-unaware workers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional
import os
from common.python.smoke import is_smoke_mode

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
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

from ch3.grace_blackwell_topology import NICInfo, discover_nics, format_cpulist


class BaselineRackPrepBenchmark(BaseBenchmark):
    """Simulates a NIC→CPU→GPU path without NUMA or IRQ steering."""

    def __init__(self):
        super().__init__()
        low_mem = is_smoke_mode()
        self.seq_len = 1024 if low_mem else 4096
        self.hidden_size = 1024 if low_mem else 4096
        self.host_batch: Optional[torch.Tensor] = None
        self.device_batch: Optional[torch.Tensor] = None
        self.norm: Optional[nn.Module] = None
        self.nic_snapshot: List[NICInfo] = []
        bytes_per_iter = self.seq_len * self.hidden_size * 4  # float32 bytes
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(bytes_per_iter),
        )

    def setup(self) -> None:
        torch.manual_seed(11)
        self.nic_snapshot = discover_nics()
        self.host_batch = torch.randn(self.seq_len, self.hidden_size, dtype=torch.float32)
        self.device_batch = torch.empty_like(self.host_batch, device=self.device)
        self.norm = nn.LayerNorm(self.hidden_size, device=self.device)
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            bytes_per_iteration=self._workload.bytes_per_iteration,
        )
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.host_batch is not None and self.device_batch is not None and self.norm is not None
        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("baseline_rack_prep", enable=enable_nvtx):
            self.device_batch.copy_(self.host_batch, non_blocking=False)
            _ = self.norm(self.device_batch)
            self._synchronize()

    def teardown(self) -> None:
        self.host_batch = None
        self.device_batch = None
        self.norm = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        low_mem = is_smoke_mode()
        # Minimum warmup=5 even in smoke mode to exclude JIT overhead
        return BenchmarkConfig(iterations=6 if low_mem else 12, warmup=5 if low_mem else 10)

    def validate_result(self) -> Optional[str]:
        if self.host_batch is None or self.norm is None:
            return "Host batch or model not initialized"
        return None

    def get_custom_metrics(self) -> Optional[dict]:
        if not self.nic_snapshot:
            return None
        summaries = [f"{n.name}:numa={n.numa_node},cpus={format_cpulist(n.local_cpus)}" for n in self.nic_snapshot]
        return {"nic_layouts": len(self.nic_snapshot), "nic_summary": ";".join(summaries)}


def get_benchmark() -> BaseBenchmark:
    return BaselineRackPrepBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=5),
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline rack prep latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    if benchmark.nic_snapshot:
        print("\nObserved NIC topology (baseline, no pinning):")
        for nic in benchmark.nic_snapshot:
            print(f"  {nic.name}: NUMA={nic.numa_node} local_cpus={format_cpulist(nic.local_cpus)} IRQs={','.join(map(str, nic.irq_ids)) or 'n/a'}")
