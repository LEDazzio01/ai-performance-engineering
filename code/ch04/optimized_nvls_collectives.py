"""optimized_nvls_collectives.py - NVLS-style NCCL collectives placeholder.

This benchmark checks for multi-GPU availability and, when possible, exercises a
small all-reduce with NCCL. If the environment lacks multiple GPUs or torchrun,
it cleanly reports SKIPPED so the harness stays green.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class NVLSCollectivesBenchmark(BaseBenchmark):
    """Tiny NCCL all-reduce meant to mirror the doc's NVLS target."""

    def __init__(self) -> None:
        super().__init__()
        self.tensor: Optional[torch.Tensor] = None
        self._initialized = False
        self._workload = WorkloadMetadata(bytes_per_iteration=0.0)

    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: NVLS collectives require >=2 GPUs")
        if not dist.is_available():
            raise RuntimeError("SKIPPED: torch.distributed not available")
        if not dist.is_initialized():
            if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
                raise RuntimeError("SKIPPED: launch with torchrun to enable NCCL NVLS demo")
            dist.init_process_group("nccl")

        os.environ.setdefault("NCCL_NVLS_ENABLE", "1")
        os.environ.setdefault("NCCL_ALGO", "Tree,Ring,NVLS")
        os.environ.setdefault("NCCL_COLLNET_ENABLE", "1")
        self.tensor = torch.ones(1024, device=self.device)
        self._initialized = True

    def benchmark_fn(self) -> Optional[dict]:
        if not self._initialized or self.tensor is None:
            raise RuntimeError("SKIPPED: NVLS benchmark not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("nvls_allreduce", enable=enable_nvtx):
            dist.all_reduce(self.tensor)
        torch.cuda.synchronize(self.device)
        return {"sum": float(self.tensor[0].item())}

    def teardown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
        super().teardown()

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

def get_benchmark() -> BaseBenchmark:
    return NVLSCollectivesBenchmark()
