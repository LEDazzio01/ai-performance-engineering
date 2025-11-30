#!/usr/bin/env python3
"""Baseline symmetric-memory perf microbench (NCCL only).

Compares simple NCCL AllReduce latency/bandwidth across payload sizes.
Use the optimized variant to see the uplift when using SymmetricMemory + direct puts.
"""
from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.metrics import compute_memory_transfer_metrics


def init_distributed() -> Tuple[int, int, int]:
    """Initialize process group for a single-node demo."""
    if not dist.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60),
        )
        torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), torch.cuda.current_device()


class BaselineSymmetricMemoryPerfBenchmark(BaseBenchmark):
    """Baseline NCCL AllReduce benchmark for symmetric memory comparison."""

    def __init__(self, size_mb: float = 16.0):
        super().__init__()
        self.size_mb = size_mb
        self.numel = int((size_mb * 1024 * 1024) / 4)  # float32
        self.tensor: Optional[torch.Tensor] = None
        self.rank = 0
        self.world_size = 1
        self._last_avg_ms = 0.0
        self._last_gbps = 0.0
        self._bytes_transferred = 0.0

    def setup(self) -> None:
        """Initialize distributed and allocate tensor."""
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: symmetric_memory_perf requires >= 2 GPUs")
        
        self.rank, self.world_size, device_id = init_distributed()
        device = torch.device("cuda", device_id)
        self.tensor = torch.ones(self.numel, device=device, dtype=torch.float32)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> Optional[Dict[str, float]]:
        """Run NCCL AllReduce and measure performance."""
        if self.tensor is None:
            raise RuntimeError("Tensor not initialized")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        dist.all_reduce(self.tensor)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        
        # Rough bytes moved: two traversals of the payload per rank (ring-style)
        bytes_moved = self.size_mb * 1024 * 1024 * 2
        gbps = (bytes_moved / (elapsed_ms / 1000.0)) / 1e9 if elapsed_ms > 0 else 0.0

        self._last_avg_ms = elapsed_ms
        self._last_gbps = gbps
        self._bytes_transferred = bytes_moved

        return {
            "allreduce.elapsed_ms": elapsed_ms,
            "allreduce.gbps": gbps,
            "allreduce.size_mb": self.size_mb,
        }

    def teardown(self) -> None:
        """Cleanup distributed resources."""
        self.tensor = None
        if dist.is_initialized():
            dist.barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        """Return memory transfer metrics for NCCL AllReduce."""
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred,
            elapsed_ms=self._last_avg_ms,
            transfer_type="nvlink",  # NCCL uses NVLink when available
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark ran successfully."""
        if self.tensor is None:
            return "Tensor not initialized"
        if self._last_avg_ms <= 0:
            return "No timing recorded"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineSymmetricMemoryPerfBenchmark()


def main() -> None:
    """Standalone execution for testing."""
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5),
    )
    benchmark = BaselineSymmetricMemoryPerfBenchmark()
    result = harness.benchmark(benchmark)

    if dist.get_rank() == 0:
        print("=" * 70)
        print("Baseline: NCCL AllReduce")
        print("=" * 70)
        print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
        print(f"Bandwidth: {benchmark._last_gbps:.2f} GB/s")


if __name__ == "__main__":
    main()
