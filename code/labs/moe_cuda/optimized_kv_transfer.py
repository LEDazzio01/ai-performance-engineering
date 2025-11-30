"""labs.moe_cuda/optimized_kv_transfer.py - Overlapped KV transfers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range


class OptimizedKVTransferBenchmark(BaseBenchmark):
    """Prefill compute + KV transfer with CUDA-stream pipelining."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 1024  # Must match baseline for valid comparison
        self.chunk_size = 256
        self.num_chunks = 16
        self.pipeline_depth = 2
        self.input_chunks: Optional[torch.Tensor] = None
        self.weight: Optional[torch.Tensor] = None
        self.workspace: Optional[torch.Tensor] = None
        self.kv_dest: Optional[torch.Tensor] = None
        self.compute_stream = torch.cuda.Stream()
        self.copy_stream = torch.cuda.Stream()
        self.events: List[torch.cuda.Event] = [
            torch.cuda.Event(enable_timing=False, blocking=False)
            for _ in range(self.pipeline_depth)
        ]
        tokens = self.num_chunks * self.chunk_size
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.num_chunks),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(0)
        self.input_chunks = torch.randn(
            self.num_chunks,
            self.chunk_size,
            self.hidden_size,
            device=self.device,
            dtype=torch.float16,
        )
        self.weight = torch.randn(self.hidden_size, self.hidden_size, device=self.device, dtype=torch.float16)
        self.workspace = torch.zeros_like(self.input_chunks)
        self.kv_dest = torch.zeros_like(self.input_chunks)
        torch.cuda.synchronize(self.device)

    def _launch_compute(self, idx: int, event: torch.cuda.Event) -> None:
        assert self.input_chunks is not None and self.weight is not None and self.workspace is not None
        with torch.cuda.stream(self.compute_stream):
            chunk = self.input_chunks[idx]  # [chunk_size, hidden]
            out = torch.matmul(chunk, self.weight)
            self.workspace[idx].copy_(out)
            event.record(self.compute_stream)

    def _launch_copy(self, idx: int, event: torch.cuda.Event) -> None:
        assert self.workspace is not None and self.kv_dest is not None
        self.copy_stream.wait_event(event)
        with torch.cuda.stream(self.copy_stream):
            self.kv_dest[idx].copy_(self.workspace[idx], non_blocking=True)

    def benchmark_fn(self) -> None:
        if any(t is None for t in (self.input_chunks, self.weight, self.workspace, self.kv_dest)):
            raise RuntimeError("Buffers not initialized")

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("moe_cuda_kv_overlap", enable=enable_nvtx):
            for i in range(self.num_chunks):
                compute_event = self.events[i % self.pipeline_depth]
                self._launch_compute(i, compute_event)
                self._launch_copy(i, compute_event)
            torch.cuda.current_stream().wait_stream(self.compute_stream)
            torch.cuda.current_stream().wait_stream(self.copy_stream)
        torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        self.input_chunks = None
        self.weight = None
        self.workspace = None
        self.kv_dest = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)  # Min warmup for CUDA

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline analysis metrics."""
        # Estimate problem size for roofline analysis
        n = getattr(self, 'N', 0) or getattr(self, 'hidden_dim', 0) or 4096
        batch = getattr(self, 'batch_size', 1) or getattr(self, 'batch', 1)
        # Simple FLOP estimate for linear layers
        flops = 2.0 * batch * n * n  # Rough estimate
        bytes_moved = batch * n * 4.0  # Input/output bytes
        arithmetic_intensity = flops / max(bytes_moved, 1.0)
        return {
            "kv_transfer.estimated_flops": flops,
            "kv_transfer.estimated_bytes": bytes_moved,
            "kv_transfer.arithmetic_intensity": arithmetic_intensity,
        }

    def validate_result(self) -> Optional[str]:
        if any(t is None for t in (self.input_chunks, self.weight, self.workspace, self.kv_dest)):
            return "Buffers not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedKVTransferBenchmark()
