"""Rack baseline: pageable staging buffers and topology-unaware workers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional
import os

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

from ch03.grace_blackwell_topology import NICInfo, discover_nics, format_cpulist


class BaselineRackPrepBenchmark(BaseBenchmark):
    """Simulates a NIC→CPU→GPU path without NUMA or IRQ steering."""

    def __init__(self):
        super().__init__()
        self.seq_len = 4096
        self.hidden_size = 4096
        self.jitter_exemption_reason = "Rack prep benchmark: fixed dimensions"
        self.host_batch: Optional[torch.Tensor] = None
        self.device_batch: Optional[torch.Tensor] = None
        self.norm: Optional[nn.Module] = None
        self.nic_snapshot: List[NICInfo] = []
        self.output: Optional[torch.Tensor] = None
        bytes_per_iter = self.seq_len * self.hidden_size * 4  # float32 bytes
        # Register workload metadata in __init__ for compliance checks
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(bytes_per_iter),
        )

    def setup(self) -> None:
        torch.manual_seed(11)
        self.nic_snapshot = discover_nics()
        self.host_batch = torch.randn(self.seq_len, self.hidden_size, dtype=torch.float32)
        self.device_batch = torch.empty_like(self.host_batch, device=self.device)
        self.norm = nn.LayerNorm(self.hidden_size, device=self.device)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.host_batch is not None and self.device_batch is not None and self.norm is not None
        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("baseline_rack_prep", enable=enable_nvtx):
            self.device_batch.copy_(self.host_batch, non_blocking=False)
            self.output = self.norm(self.device_batch)
            self._synchronize()

    def teardown(self) -> None:
        self.host_batch = None
        self.device_batch = None
        self.norm = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=12, warmup=10)

    def validate_result(self) -> Optional[str]:
        if self.host_batch is None or self.norm is None:
            return "Host batch or model not initialized"
        return None

    def get_custom_metrics(self) -> Optional[dict]:
        if not self.nic_snapshot:
            return None
        summaries = [f"{n.name}:numa={n.numa_node},cpus={format_cpulist(n.local_cpus)}" for n in self.nic_snapshot]
        return {"nic_layouts": len(self.nic_snapshot), "nic_summary": ";".join(summaries)}

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is not None:
            return self.output.detach().clone()
        return torch.tensor([0.0], dtype=torch.float32, device=self.device)
    
    def get_output_tolerance(self) -> tuple:
        """Return custom tolerance for LayerNorm output comparison."""
        return (1e-4, 1e-4)

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"seq_len": self.seq_len, "hidden_size": self.hidden_size}


def get_benchmark() -> BaseBenchmark:
    return BaselineRackPrepBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
