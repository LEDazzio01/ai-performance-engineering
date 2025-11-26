"""baseline_cluster_multicast.py - DSMEM-less cluster multicast baseline.

This is a lightweight, single-GPU stand-in for the cluster multicast baseline
described in the docs. It runs a couple of matmuls to approximate traffic
without requiring CTA clusters or DSMEM so the harness has a runnable target.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class BaselineClusterMulticastBenchmark(BaseBenchmark):
    """Runs two sequential matmuls to mimic non-multicast traffic."""

    def __init__(self) -> None:
        super().__init__()
        self.linear_a: Optional[nn.Linear] = None
        self.linear_b: Optional[nn.Linear] = None
        self.inputs: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(bytes_per_iteration=0.0)

    def setup(self) -> None:
        torch.manual_seed(0)
        hidden = 4096
        self.linear_a = nn.Linear(hidden, hidden, bias=False).to(self.device)
        self.linear_b = nn.Linear(hidden, hidden, bias=False).to(self.device)
        self.inputs = torch.randn(256, hidden, device=self.device, dtype=torch.bfloat16)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Optional[dict]:
        if self.linear_a is None or self.linear_b is None or self.inputs is None:
            raise RuntimeError("SKIPPED: multicast baseline not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        start = self._record_start()
        with nvtx_range("cluster_multicast_baseline", enable=enable_nvtx):
            x = self.linear_a(self.inputs)
            x = torch.relu(x)
            x = self.linear_b(x)
        torch.cuda.synchronize(self.device)
        latency_ms = self._record_stop(start)
        return {"latency_ms": latency_ms}

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )
}

def get_benchmark() -> BaseBenchmark:
    return BaselineClusterMulticastBenchmark()
