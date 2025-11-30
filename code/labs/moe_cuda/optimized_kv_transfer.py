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
        import gc
        
        # Clean up any leftover CUDA graph state from previous benchmarks
        # to prevent "Offset increment outside graph capture" errors
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        try:
            if hasattr(torch.cuda, 'graph_pool_trim'):
                torch.cuda.graph_pool_trim()
        except Exception:
            pass
        
        # CRITICAL: Reset CUDA random number generator state
        # CUDA graphs capture the RNG offset, which causes "Offset increment 
        # outside graph capture" errors when using torch.randn
        try:
            device_idx = torch.cuda.current_device()
            gen = torch.cuda.default_generators[device_idx]
            # set_offset(0) properly resets the graph capture state
            gen.set_offset(0)
            gen.manual_seed(0)
        except Exception:
            pass
        
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        
        try:
            torch._inductor.cudagraph_trees.reset_cudagraph_trees()
        except Exception:
            pass
        
        torch.manual_seed(0)
        # Create tensors using CPU randn + to(device) to avoid CUDA RNG graph capture issues
        # This is safer because CPU randn doesn't interact with CUDA graph capture state
        self.input_chunks = torch.randn(
            self.num_chunks,
            self.chunk_size,
            self.hidden_size,
            dtype=torch.float16,
        ).to(self.device)
        self.weight = torch.randn(self.hidden_size, self.hidden_size, dtype=torch.float16).to(self.device)
        self.workspace = torch.zeros_like(self.input_chunks)
        self.kv_dest = torch.zeros_like(self.input_chunks)
        
        # Warmup the streams to avoid capture issues
        with torch.cuda.stream(self.compute_stream):
            _ = torch.matmul(self.input_chunks[0], self.weight)
        with torch.cuda.stream(self.copy_stream):
            self.kv_dest[0].copy_(self.workspace[0])
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

    def get_input_signature(self) -> dict:
        """Return workload signature for verification.
        
        Only hidden_size is compared since baseline/optimized use different
        batching strategies (sequential vs pipelined) by design.
        """
        return {
            "hidden_size": self.hidden_size,
            "dtype": "float16",
        }


def get_benchmark() -> BaseBenchmark:
    return OptimizedKVTransferBenchmark()
