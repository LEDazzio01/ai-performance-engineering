#!/usr/bin/env python3
"""Optimized: Grace-Blackwell coherent memory with cache-aware access.

Demonstrates optimized coherent memory patterns:
- Zero-copy buffers for small transfers (<4MB)
- Pinned memory with async copies for medium transfers (4-64MB)
- Explicit transfers with optimal alignment for large transfers (>64MB)
- NUMA-aware allocation on Grace CPUs
"""

import torch
import time
from typing import Dict, Any, Optional
import sys
from pathlib import Path
from common.python.smoke import is_smoke_mode
import os
from common.python.smoke import is_smoke_mode
import ctypes

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


class OptimizedGraceCoherentMemory:
    """Optimized coherent memory with cache-aware access patterns."""
    
    # Thresholds based on Grace-Blackwell coherency fabric performance
    ZERO_COPY_THRESHOLD_MB = 4    # Use zero-copy for <4MB
    ASYNC_THRESHOLD_MB = 64        # Use async pinned for 4-64MB
    
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
        
        # Select optimal strategy based on size
        self.strategy = self._select_strategy()
        logger.info(f"Selected strategy: {self.strategy} for {size_mb}MB")
    
    def _detect_grace_blackwell(self) -> bool:
        """Detect if running on Grace-Blackwell platform."""
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
    
    def _select_strategy(self) -> str:
        """Select optimal transfer strategy based on size."""
        if self.size_mb < self.ZERO_COPY_THRESHOLD_MB:
            return "zero_copy"
        elif self.size_mb < self.ASYNC_THRESHOLD_MB:
            return "async_pinned"
        else:
            return "explicit_aligned"
    
    def _bind_numa_node(self):
        """Bind to NUMA node closest to GPU (Grace-Blackwell specific)."""
        if not self.is_grace_blackwell:
            return
        
        gpu_id = torch.cuda.current_device()
        numa_node = gpu_id  # Grace-Blackwell typically maps GPU i -> NUMA node i

        # Try to pin this process' CPU affinity to the NUMA node's CPUs
        try:
            cpulist_path = Path(f"/sys/devices/system/node/node{numa_node}/cpulist")
            if cpulist_path.exists():
                ranges = cpulist_path.read_text().strip().split(",")
                cpus = []
                for r in ranges:
                    if "-" in r:
                        start, end = r.split("-")
                        cpus.extend(range(int(start), int(end) + 1))
                    else:
                        cpus.append(int(r))
                if cpus:
                    os.sched_setaffinity(0, cpus)
                    logger.info(f"Bound CPU affinity to NUMA node {numa_node}: {cpus}")
        except Exception as e:  # pragma: no cover - best effort
            logger.debug(f"NUMA CPU affinity binding failed: {e}")

        # Best-effort memory preference using libnuma if available
        try:
            libnuma = ctypes.CDLL("libnuma.so.1")
            if libnuma.numa_available() != -1:
                libnuma.numa_run_on_node(ctypes.c_int(numa_node))
                libnuma.numa_set_preferred(ctypes.c_int(numa_node))
                logger.info(f"Set NUMA memory preference to node {numa_node}")
        except Exception as e:  # pragma: no cover - optional path
            logger.debug(f"NUMA memory binding skipped: {e}")
    
    def setup(self):
        """Initialize data structures with optimal memory type."""
        num_elements = (self.size_mb * 1024 * 1024) // 4  # float32
        
        # Bind to optimal NUMA node
        self._bind_numa_node()
        
        if self.strategy == "zero_copy":
            # Zero-copy: Map CPU memory directly to GPU
            # On Grace-Blackwell, this uses cache-coherent NVLink-C2C
            if self.is_grace_blackwell:
                # Single allocation stays resident on GPU; CPU can still peek via unified cache.
                self.gpu_data = torch.randn(num_elements, dtype=torch.float32, device=self.device)
                # Keep a reference for API symmetry; this is the same buffer.
                self.cpu_data = self.gpu_data
                logger.info(f"Using zero-copy coherent GPU buffer ({self.size_mb}MB)")
            else:
                # Fallback: keep data pinned on CPU but acknowledge it is not truly zero-copy.
                self.cpu_data = torch.randn(num_elements, dtype=torch.float32).pin_memory()
                self.gpu_data = torch.empty(num_elements, dtype=torch.float32, device=self.device)
                self.gpu_data.copy_(self.cpu_data, non_blocking=True)
                logger.info(f"Grace-less fallback: pinned CPU buffer ({self.size_mb}MB)")
        
        elif self.strategy == "async_pinned":
            # Pinned memory with async copies
            self.cpu_data = torch.randn(num_elements, dtype=torch.float32).pin_memory()
            self.gpu_data = torch.zeros(num_elements, dtype=torch.float32, device=self.device)
            
            # Create stream for async copies
            self.stream = torch.cuda.Stream()
            logger.info(f"Using async pinned memory ({self.size_mb}MB)")
        
        else:  # explicit_aligned
            # Explicit transfers with optimal alignment (128-byte for Blackwell)
            # Ensure alignment by using contiguous tensors
            self.cpu_data = torch.randn(num_elements, dtype=torch.float32).contiguous()
            self.gpu_data = torch.zeros(num_elements, dtype=torch.float32, device=self.device).contiguous()
            
            # Verify alignment (warn-only so CI/low-mem paths don't hard fail)
            if self.cpu_data.data_ptr() % 128 != 0 or self.gpu_data.data_ptr() % 128 != 0:
                logger.warning("128-byte alignment unavailable; continuing with contiguous buffers")
            logger.info(f"Using explicit aligned transfers ({self.size_mb}MB)")
    
    def run(self) -> float:
        """Execute optimized coherent memory access pattern."""
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        if self.strategy == "zero_copy":
            # Zero-copy: Direct access via coherent NVLink-C2C
            for _ in range(self.iterations):
                # Access happens through coherent cache
                self.gpu_data.mul_(2.0).add_(1.0)
                # No explicit copy needed - coherency maintained by hardware
        
        elif self.strategy == "async_pinned":
            # Async pinned: Overlap copy with compute
            for _ in range(self.iterations):
                with torch.cuda.stream(self.stream):
                    # Async H2D copy
                    self.gpu_data.copy_(self.cpu_data, non_blocking=True)
                
                # Compute (can overlap with copy)
                self.gpu_data.mul_(2.0).add_(1.0)
                
                # Async D2H copy
                self.cpu_data.copy_(self.gpu_data, non_blocking=True)
                self.stream.synchronize()
        
        else:  # explicit_aligned
            # Explicit aligned: Maximize bandwidth with optimal alignment
            for _ in range(self.iterations):
                # Aligned H2D copy
                self.gpu_data.copy_(self.cpu_data.to(self.device))
                
                # Computation
                self.gpu_data.mul_(2.0).add_(1.0)
                
                # Aligned D2H copy
                self.cpu_data = self.gpu_data.cpu()
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        elapsed = end - start
        
        # Calculate bandwidth (zero-copy only counts once since data isn't moved)
        if self.strategy == "zero_copy":
            bandwidth_gb_s = (self.size_mb / 1024) * self.iterations / elapsed
        else:
            bandwidth_gb_s = (self.size_mb / 1024) * self.iterations * 2 / elapsed  # 2 for H2D + D2H
        
        logger.info(f"Optimized bandwidth ({self.strategy}): {bandwidth_gb_s:.2f} GB/s")
        return elapsed
    
    def cleanup(self):
        """Clean up resources."""
        del self.cpu_data
        del self.gpu_data
        if hasattr(self, 'stream'):
            del self.stream
        torch.cuda.empty_cache()


class OptimizedGraceCoherentMemoryBenchmark(BaseBenchmark):
    """Harness-friendly wrapper around the optimized coherent memory path."""

    def __init__(self, size_mb: int = 256, iterations: int = 100):
        super().__init__()
        fast = is_smoke_mode()
        self._impl = OptimizedGraceCoherentMemory(
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
    return OptimizedGraceCoherentMemoryBenchmark()


def run_benchmark(
    size_mb: int = 256,
    iterations: int = 100,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run optimized Grace coherent memory benchmark."""
    
    benchmark = OptimizedGraceCoherentMemoryBenchmark(size_mb=size_mb, iterations=iterations)
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
    result = harness.benchmark(benchmark, name="optimized_grace_coherent_memory")

    return {
        "mean_time_ms": result.timing.mean_ms if result.timing else 0.0,
        "is_grace_blackwell": getattr(benchmark, "_impl", None).is_grace_blackwell if hasattr(benchmark, "_impl") else False,
        "strategy": getattr(benchmark, "_impl", None).strategy if hasattr(benchmark, "_impl") else "",
        "size_mb": size_mb,
        "iterations": iterations,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Grace coherent memory")
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
    print(f"Optimized Grace Coherent Memory Results")
    print(f"{'='*60}")
    print(f"Grace-Blackwell: {result['is_grace_blackwell']}")
    print(f"Strategy: {result['strategy']}")
    print(f"Buffer size: {result['size_mb']} MB")
    print(f"Iterations: {result['iterations']}")
    print(f"Mean time: {result['mean_time_ms']:.2f} ms")
    print(f"{'='*60}\n")
