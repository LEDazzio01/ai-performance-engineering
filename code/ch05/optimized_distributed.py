"""optimized_distributed.py - Optimized distributed operations in storage I/O context."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.gpu_requirements import skip_if_insufficient_gpus


class OptimizedDistributedBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Distributed sum across ranks; falls back to single-GPU when world_size=1."""
    
    def __init__(self):
        super().__init__()
        self.data: Optional[torch.Tensor] = None
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        self.N = 10_000_000
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N),
        )
    
    def setup(self) -> None:
        """Setup: Initialize data and (optional) distributed process group."""
        skip_if_insufficient_gpus()
        torch.manual_seed(42)
        
        # Initialize distributed if running in multi-rank mode
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            if world_size > 1:
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    rank=rank,
                    world_size=world_size,
                )
                self.is_distributed = True
                self.rank = rank
                self.world_size = world_size

        local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
        
        chunk_size = self.N // max(self.world_size, 1)
        start_idx = self.rank * chunk_size
        end_idx = start_idx + chunk_size if self.rank < self.world_size - 1 else self.N
        self.data = torch.randn(end_idx - start_idx, device=self.device)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: distributed reduction when enabled."""
        assert self.data is not None
        with self._nvtx_range("optimized_distributed"):
            local_result = self.data.sum()
            if self.is_distributed:
                dist.all_reduce(local_result, op=dist.ReduceOp.SUM)
                result = local_result / self.world_size
            else:
                result = local_result
            _ = result
            self._synchronize()
        self.output = result.detach().clone()

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={
                "data": self.data,
                "world_size": torch.tensor([self.world_size], device=self.device),
            },
            output=self.output,
            batch_size=self.data.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.1, 1.0),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        if self.is_distributed:
            dist.destroy_process_group()
        self.data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_storage_io_metrics
        return compute_storage_io_metrics(
            bytes_read=getattr(self, '_bytes_read', 0.0),
            bytes_written=getattr(self, '_bytes_written', 0.0),
            read_time_ms=getattr(self, '_read_time_ms', 1.0),
            write_time_ms=getattr(self, '_write_time_ms', 1.0),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedDistributedBenchmark()
