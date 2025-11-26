#!/usr/bin/env python3
"""Baseline: Grace-Blackwell coherent memory without optimization.

Demonstrates basic coherent memory access patterns on Grace-Blackwell systems
without cache-aware optimizations or NUMA awareness.
"""

import torch
import time
from typing import Dict, Any, Optional
import sys
from pathlib import Path
import os
from common.python.smoke import is_smoke_mode

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkHarness,
    BenchmarkConfig,
    BenchmarkMode,
    ExecutionMode,
    WorkloadMetadata,
)
from common.python.logger import get_logger

logger = get_logger(__name__)


class BaselineGraceCoherentMemory:
    """Baseline coherent memory access without optimization."""
    
    def __init__(self, size_mb: int = 256, iterations: int = 100):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Grace coherent memory benchmark")
        self.size_mb = size_mb
        self.iterations = iterations
        self.device = torch.device("cuda")
        
        # Check if we're on Grace-Blackwell
        self.is_grace_blackwell = self._detect_grace_blackwell()
        if not self.is_grace_blackwell:
            logger.warning("Not running on Grace-Blackwell; using fallback path")
    
    def _detect_grace_blackwell(self) -> bool:
        """Detect if running on Grace-Blackwell platform."""
        if not torch.cuda.is_available():
            return False
        
        try:
            props = torch.cuda.get_device_properties(0)
            # GB200/GB300 has compute capability 12.1
            if props.major == 12 and props.minor == 1:
                # Additional check for Grace CPU (ARM architecture)
                import platform
                if platform.machine() in ['aarch64', 'arm64']:
                    return True
        except Exception as e:
            logger.debug(f"Grace-Blackwell detection failed: {e}")
        
        return False
    
    def setup(self):
        """Initialize data structures with pageable CPU memory (baseline)."""
        num_elements = (self.size_mb * 1024 * 1024) // 4  # float32
        
        # Baseline: Use regular pageable memory without pinning
        # This will go through explicit H2D transfers
        self.cpu_data = torch.randn(num_elements, dtype=torch.float32)
        
        # GPU buffer for computation
        self.gpu_data = torch.zeros(num_elements, dtype=torch.float32, device=self.device)
        
        logger.info(f"Allocated {self.size_mb}MB pageable CPU memory")
    
    def run(self) -> float:
        """Execute baseline coherent memory access pattern."""
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(self.iterations):
            # Baseline: Explicit H2D copy
            self.gpu_data.copy_(self.cpu_data.to(self.device))
            
            # Simple computation
            self.gpu_data.mul_(2.0).add_(1.0)
            
            # Baseline: Explicit D2H copy
            self.cpu_data = self.gpu_data.cpu()
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        elapsed = end - start
        bandwidth_gb_s = (self.size_mb / 1024) * self.iterations * 2 / elapsed  # 2 for H2D + D2H
        
        logger.info(f"Baseline bandwidth: {bandwidth_gb_s:.2f} GB/s")
        return elapsed
    
    def cleanup(self):
        """Clean up resources."""
        del self.cpu_data
        del self.gpu_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class GraceCoherentMemoryBenchmark(BaseBenchmark):
    """Harness-friendly wrapper around the baseline coherent memory example."""

    def __init__(self, size_mb: int = 256, iterations: int = 100):
        super().__init__()
        fast = is_smoke_mode()
        self._impl = BaselineGraceCoherentMemory(
            size_mb=size_mb,
            iterations=20 if fast else iterations,
        )
        bytes_per_iter = size_mb * 1024 * 1024 * 2  # H2D + D2H
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(bytes_per_iter),
        )
        self.elapsed_s: Optional[float] = None
        self.bandwidth_gb_s: Optional[float] = None

    def setup(self) -> None:
        self._impl.setup()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            bytes_per_iteration=self._workload.bytes_per_iteration,
        )

    def benchmark_fn(self) -> None:
        elapsed = self._impl.run()
        self.elapsed_s = elapsed
        self.bandwidth_gb_s = (self._impl.size_mb / 1024) * self._impl.iterations * 2 / elapsed

    def teardown(self) -> None:
        self._impl.cleanup()
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=0, enable_memory_tracking=False)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return memory transfer metrics for grace_coherent_memory."""
        from common.python.benchmark_metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self.size,
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        if self.elapsed_s is None:
            return "Benchmark did not run"
        return None


def get_benchmark() -> BaseBenchmark:
    return GraceCoherentMemoryBenchmark()


def run_benchmark(
    size_mb: int = 256,
    iterations: int = 100,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run baseline Grace coherent memory benchmark."""
    
    benchmark = GraceCoherentMemoryBenchmark(size_mb=size_mb, iterations=iterations)
    harness = BenchmarkHarness(
        mode=BenchmarkMode.TRAINING,
        config=BenchmarkConfig(
            iterations=1,
            warmup=0,
            profile_mode=profile,
            use_subprocess=False,
            execution_mode=ExecutionMode.THREAD,
        ),
    )
    result = harness.benchmark(benchmark, name="baseline_grace_coherent_memory")

    return {
        "mean_time_ms": result.timing.mean_ms if result.timing else 0.0,
        "bandwidth_gb_s": (benchmark.bandwidth_gb_s or 0.0) if hasattr(benchmark, "bandwidth_gb_s") else 0.0,
        "is_grace_blackwell": getattr(benchmark, "_impl", None).is_grace_blackwell if hasattr(benchmark, "_impl") else False,
        "size_mb": size_mb,
        "iterations": iterations,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline Grace coherent memory")
    parser.add_argument("--size-mb", type=int, default=256, help="Buffer size in MB")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--profile", type=str, default="none", help="Profiling mode")
    
    args = parser.parse_args()
    
    result = run_benchmark(
        size_mb=args.size_mb,
        iterations=args.iterations,
        profile=args.profile,
    )
    
    print(f"\n{'='*60}")
    print(f"Baseline Grace Coherent Memory Results")
    print(f"{'='*60}")
    print(f"Grace-Blackwell: {result['is_grace_blackwell']}")
    print(f"Buffer size: {result['size_mb']} MB")
    print(f"Iterations: {result['iterations']}")
    print(f"Mean time: {result['mean_time_ms']:.2f} ms")
    print(f"{'='*60}\n")
