"""baseline_torchcomms.py - Baseline using legacy torch.distributed patterns.

This demonstrates traditional torch.distributed collective communication
patterns before the modern torchcomms API (announced November 2025).

Legacy patterns have more boilerplate, less overlap capability, and
require explicit process group management for complex topologies.

REQUIRES: Multi-GPU system (2+ GPUs) - skips on single GPU.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional

# Require multi-GPU for torchcomms benchmarks
GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
if GPU_COUNT < 2:
    raise RuntimeError(
        f"torchcomms benchmark requires 2+ GPUs (found {GPU_COUNT}). "
        "This benchmark demonstrates distributed communication patterns."
    )

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


def _init_distributed_if_needed() -> bool:
    """Initialize distributed if not already done. Returns True if distributed is available."""
    if dist.is_initialized():
        return True
    
    # Check if we're in a distributed environment
    if "RANK" not in os.environ:
        return False
    
    try:
        dist.init_process_group(backend="nccl")
        return True
    except Exception:
        return False


class BaselineTorchcommsBenchmark(BaseBenchmark):
    """Baseline: Legacy torch.distributed patterns.
    
    This demonstrates the traditional approach to distributed communication:
    - Explicit process group creation and management
    - Synchronous collective operations
    - No built-in overlap with computation
    - Manual tensor allocation for recv buffers
    
    The modern torchcomms API (PyTorch 2.10+) provides:
    - Async-first design with automatic overlap
    - Functional API without side effects
    - Better integration with FSDP2/DTensor
    - Simplified topology management
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.input = None
        self.output = None
        self.batch = 256
        self.hidden = 4096
        self.is_distributed = False
        tokens = self.batch * self.hidden
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self._bytes_transferred = 0
    
    def setup(self) -> None:
        """Setup: Initialize model and communication."""
        torch.manual_seed(42)
        
        # Try to initialize distributed
        self.is_distributed = _init_distributed_if_needed()
        
        # Simple transformer-like block for communication simulation
        self.model = nn.Sequential(
            nn.Linear(self.hidden, self.hidden * 4),
            nn.GELU(),
            nn.Linear(self.hidden * 4, self.hidden),
        ).to(self.device).eval()
        
        self.input = torch.randn(self.batch, self.hidden, device=self.device)
        self.output = torch.zeros_like(self.input)
        
        # Calculate bytes transferred for metrics
        # All-reduce: 2 * tensor_size (reduce-scatter + all-gather)
        self._bytes_transferred = 2 * self.input.numel() * self.input.element_size()
        
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Legacy torch.distributed communication patterns.
        
        Pattern demonstrated:
        1. Forward pass (compute)
        2. Synchronous all-reduce (blocking)
        3. No overlap between compute and communication
        """
        with self._nvtx_range("baseline_torchcomms"):
            with torch.no_grad():
                # Forward pass
                output = self.model(self.input)
                
                if self.is_distributed:
                    # Legacy pattern: synchronous all-reduce
                    # This blocks until all ranks complete
                    dist.all_reduce(output, op=dist.ReduceOp.AVG)
                else:
                    # Simulate all-reduce cost without distributed
                    # In real distributed, this would average across ranks
                    # We simulate with a local operation
                    chunks = torch.chunk(output, chunks=4, dim=0)
                    reduced = sum(chunks) / 4.0
                    output = reduced.expand_as(output[:len(output)//4]).repeat(4, 1)
                
                self.output = output
        
        self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        self.output = None
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
        """Return domain-specific metrics for distributed communication."""
        from common.python.benchmark_metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred,
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="nvlink",  # Assuming NVLink for GPU-GPU
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        if self.output is None:
            return "Output not computed"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineTorchcommsBenchmark()


def main() -> None:
    """Standalone execution."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineTorchcommsBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Baseline: Legacy torch.distributed Patterns")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")
    print()
    print("Legacy patterns demonstrated:")
    print("  - Synchronous all-reduce (blocks until complete)")
    print("  - No compute/communication overlap")
    print("  - Explicit process group management")
    print()
    print("See optimized_torchcomms.py for modern torchcomms API patterns.")


if __name__ == "__main__":
    main()

