"""Optimized graph execution with CUDA graph replay.

This module demonstrates CUDA graph replay benefits - eliminating kernel
launch overhead by pre-compiling the execution graph.

Key Features:
- CUDA graph capture and replay
- Eliminates per-kernel launch overhead
- Batches multiple operations into single submission

For conditional graph nodes (CUDA 12.4+), see the CUDA binary implementations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from torch.cuda import CUDAGraph

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


def supports_conditional_graphs() -> bool:
    """Check if conditional graph nodes are supported (CUDA 12.4+)."""
    if not torch.cuda.is_available():
        return False
    cuda_version = torch.version.cuda
    if cuda_version:
        try:
            version_num = int(cuda_version.replace(".", "")[:3])
            return version_num >= 124
        except (ValueError, TypeError):
            pass
    return False


class OptimizedGraphBenchmark(BaseBenchmark):
    """Optimized benchmark using CUDA graph replay.
    
    CUDA graphs eliminate kernel launch overhead by:
    1. Capturing the execution pattern once
    2. Replaying it with a single GPU submission
    
    This is ~10-100x faster for launch-bound workloads.
    """
    
    def __init__(self):
        super().__init__()
        self.batch_size = 32
        self.seq_len = 512
        self.hidden_dim = 2048
        
        self.data: Optional[torch.Tensor] = None
        self._graph: Optional[CUDAGraph] = None
        self._graph_stream: Optional[torch.cuda.Stream] = None
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def _compute_ops(self) -> None:
        """Operations to capture in graph - uses in-place ops only."""
        # All in-place operations for stable tensor addresses
        self.data.mul_(0.99)
        for _ in range(16):  # 16 in-place ops = 16 kernel launches normally
            self.data.add_(0.001)
            self.data.mul_(1.0001)
        self.data.relu_()
        self.data.mul_(1.001)
    
    def setup(self) -> None:
        """Setup CUDA graph with static buffers."""
        torch.manual_seed(42)
        
        dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        
        # Static buffer - address must not change
        self.data = torch.randn(
            self.batch_size, self.seq_len, self.hidden_dim,
            device=self.device, dtype=dtype
        )
        
        # Non-default stream required for graph capture
        self._graph_stream = torch.cuda.Stream()
        
        # Warmup before capture (required for some ops)
        torch.cuda.synchronize()
        with torch.cuda.stream(self._graph_stream):
            self._compute_ops()
        torch.cuda.synchronize()
        
        # Reset data for graph capture
        self.data.fill_(1.0)
        
        # Capture the execution graph
        self._graph = CUDAGraph()
        with torch.cuda.stream(self._graph_stream):
            with torch.cuda.graph(self._graph, stream=self._graph_stream):
                self._compute_ops()
        
        # Warmup graph replay
        for _ in range(5):
            self._graph.replay()
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark CUDA graph replay - much faster than fresh launches."""
        with self._nvtx_range("graph_replay"):
            self._graph.replay()
        self._synchronize()
    
    def teardown(self) -> None:
        """Clean up."""
        self.data = None
        self._graph = None
        self._graph_stream = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=100,
            warmup=20,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return graph metrics using standard helpers."""
        from common.python.benchmark_metrics import compute_graph_metrics
        
        # Estimate launch overhead (typical kernel launch is 5-10us)
        baseline_launch_us = 8.0 * 35  # ~35 ops × 8us each
        graph_launch_us = 15.0  # Single graph launch overhead
        
        metrics = compute_graph_metrics(
            baseline_launch_overhead_us=baseline_launch_us,
            graph_launch_overhead_us=graph_launch_us,
            num_nodes=35,
            num_iterations=100,
        )
        metrics["graph.uses_cuda_graph"] = 1.0
        metrics["graph.conditional_support"] = 1.0 if supports_conditional_graphs() else 0.0
        return metrics


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedGraphBenchmark()


def main() -> None:
    """Standalone execution."""
    print("=" * 70)
    print("Optimized: CUDA Graph Replay")
    print("=" * 70)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        major, minor = torch.cuda.get_device_capability()
        print(f"Compute Capability: {major}.{minor}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Conditional Graphs (12.4+): {supports_conditional_graphs()}")
    
    print()
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=100, warmup=20)
    )
    
    benchmark = get_benchmark()
    result = harness.benchmark(benchmark)
    
    print(f"Results:")
    print(f"  Average: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"  Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print()
    print("Optimizations:")
    print("  ✓ CUDA graph capture and replay")
    print("  ✓ ~35 kernel launches batched into single submission")
    print("  ✓ Eliminates per-kernel launch overhead")
    print("  ✓ Ideal for inference with fixed shapes")


if __name__ == "__main__":
    main()
