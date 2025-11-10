"""baseline_storage_cpu.py - CPU-mediated storage I/O (baseline).

Traditional approach: Storage → CPU → GPU (double copy overhead).
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
import tempfile
import os
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import numpy as np

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch5")
    return torch.device("cuda")


class BaselineStorageCpuBenchmark(Benchmark):
    """CPU-mediated I/O - double copy overhead."""
    
    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.filepath = None
        self.size_mb = 64  # Smaller for faster benchmark
        self.size = self.size_mb * 1024 * 1024 // 4  # float32
    
    def setup(self) -> None:
        """Setup: Initialize data and create temp file."""
        torch.manual_seed(42)
        self.data = torch.randn(self.size, device=self.device, dtype=torch.float32)
        
        # Create temp file
        f = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
        self.filepath = f.name
        f.close()
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: CPU-mediated I/O."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("storage_cpu", enable=enable_nvtx):
            # Write: GPU → CPU → Storage
            cpu_data = self.data.cpu().numpy()
            np.save(self.filepath, cpu_data)
            
            # Read: Storage → CPU → GPU
            cpu_loaded = np.load(self.filepath)
            self.data = torch.from_numpy(cpu_loaded).to(self.device)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        if self.filepath and os.path.exists(self.filepath):
            os.unlink(self.filepath)
        self.data = None
        self.filepath = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data tensor not initialized"
        if self.data.shape[0] != self.size:
            return f"Data size mismatch: expected {self.size}, got {self.data.shape[0]}"
        if not torch.isfinite(self.data).all():
            return "Data contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineStorageCpuBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Storage CPU: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")

