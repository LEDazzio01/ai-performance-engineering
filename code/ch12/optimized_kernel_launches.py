"""optimized_kernel_launches.py - CUDA Graphs optimization.

Demonstrates CUDA Graphs to reduce kernel launch overhead.
All operations captured in single graph → single launch overhead.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch12")
    return torch.device("cuda")


class OptimizedKernelLaunchesBenchmark(Benchmark):
    """Benchmark implementation following Benchmark protocol."""
    
    def __init__(self):
        self.device = resolve_device()
        self.x_template = None
        self.x_capture = None
        self.graph = None
        self.replay_fn = None
        self.size = (1024, 1024)
        self.iterations = 1000
    
    def setup(self) -> None:
        """Setup: initialize tensor and capture CUDA graph."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        # Use bfloat16 for GPU performance
        dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.x_template = torch.randn(*self.size, device=self.device, dtype=dtype)
        
        # Warmup before graph capture
        for _ in range(10):
            x_warmup = self.x_template.clone()
            for _ in range(self.iterations):
                x_warmup = x_warmup + 1.0
                x_warmup = x_warmup * 0.99
                x_warmup = torch.relu(x_warmup)
        torch.cuda.synchronize()
        
        # Capture graph
        self.graph = torch.cuda.CUDAGraph()
        self.x_capture = self.x_template.clone()
        with torch.cuda.graph(self.graph):
            for _ in range(self.iterations):
                self.x_capture = self.x_capture + 1.0
                self.x_capture = self.x_capture * 0.99
                self.x_capture = torch.relu(self.x_capture)
        
        # Create replay function
        def replay():
            self.graph.replay()
            return self.x_capture
        
        self.replay_fn = replay
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("kernel_launches", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.replay_fn()

    def teardown(self) -> None:
        """Cleanup."""
        del self.x_template, self.x_capture, self.graph, self.replay_fn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.x_template is None:
            return "Input tensor x_template not initialized"
        if self.graph is None:
            return "CUDA graph not initialized"
        if self.replay_fn is None:
            return "Replay function not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedKernelLaunchesBenchmark()


def main() -> None:
    """Standalone execution with timing."""
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=10)
    )
    benchmark = OptimizedKernelLaunchesBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Optimized: CUDA Graphs (Single Launch)")
    print("=" * 70)
    print(f"Tensor size: {benchmark.size}")
    print(f"Operations per iteration: 3 (add, multiply, relu)")
    print(f"Total operations: {benchmark.iterations * 3}")
    print("Optimization: All captured in single graph → 1 launch overhead\n")
    
    baseline_launches = benchmark.iterations * 3
    optimized_launches = 1
    
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")
    print(f"Kernel launches per run: {optimized_launches} (vs {baseline_launches} in baseline)")
    print(f"Launch overhead reduction: {baseline_launches / optimized_launches:.0f}x")
    print(f"Speedup: Reduced launch overhead enables better GPU utilization")


if __name__ == "__main__":
    main()
