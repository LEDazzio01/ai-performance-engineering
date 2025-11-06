"""optimized_performance_graphs.py - Optimized performance benchmark with CUDA Graphs.

Demonstrates CUDA Graphs optimization to reduce kernel launch overhead.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

try:
    import ch1.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from common.python.compile_utils import compile_model

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return a usable device, falling back to CPU if CUDA is unavailable or unsupported."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1, device="cuda")
        return torch.device("cuda")
    except Exception as exc:
        print(f"WARNING: CUDA unavailable or unsupported ({exc}); falling back to CPU.")
        return torch.device("cpu")


class OptimizedPerformanceGraphsBenchmark(Benchmark):
    """Benchmark implementation with CUDA Graphs optimization."""
    
    def __init__(self, batch_size: int = 128):
        self.device = resolve_device()
        self.batch_size = batch_size
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:
            model = torch.compile(None, mode="reduce-overhead", backend="inductor")
        except Exception:
            pass  # Fallback to eager if compilation fails

        # Optimization: Compile model for kernel fusion and optimization
        try:
            self.model = torch.compile(None, mode="reduce-overhead", backend="inductor")
        except Exception:
            pass  # Fallback to eager if compilation fails

        self.data_buf = None
        self.target_buf = None
        self.host_data = None
        self.host_target = None
        self.graph = None
    
    def setup(self) -> None:
        """Setup: initialize model and capture CUDA graph."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if self.device.type != "cuda":
            raise RuntimeError("CUDA Graphs require CUDA device")
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
        ).to(self.device)
        
        self.model.eval()
        
        # Preallocated device buffers
        self.data_buf = torch.empty(self.batch_size, 256, device=self.device)
        self.target_buf = torch.empty(self.batch_size, dtype=torch.long, device=self.device)
        
        # Pinned host staging
        self.host_data = torch.empty(self.batch_size, 256, pin_memory=True)
        self.host_target = torch.empty(self.batch_size, dtype=torch.long, pin_memory=True)
        
        # CUDA Graphs: Capture the forward pass
        torch.cuda.synchronize(self.device)
        torch.set_grad_enabled(False)
        
        # Initialize target values for warmup (must be in valid range [0, 9])
        self.target_buf.random_(0, 10).clamp_(0, 9)
        
        for _ in range(5):  # Warmup
            logits = self.model(self.data_buf)
            loss = torch.nn.functional.cross_entropy(logits, self.target_buf)
        torch.cuda.synchronize(self.device)
        
        # Ensure targets are valid before graph capture
        self.target_buf.random_(0, 10).clamp_(0, 9)
        
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            logits = self.model(self.data_buf)
            loss = torch.nn.functional.cross_entropy(logits, self.target_buf)
        torch.cuda.synchronize(self.device)
        torch.set_grad_enabled(True)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark with CUDA graph replay."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_performance_graphs", enable=enable_nvtx):
            self.host_data.normal_(0, 1)
            # Generate targets in valid range [0, 9] for 10-class classification
            # Use randint to ensure values are in [0, 9] inclusive
            self.host_target.random_(0, 10)  # Generates [0, 10) = [0, 9]
            # Clamp to ensure no values >= 10 (safety check)
            self.host_target.clamp_(0, 9)
            self.data_buf.copy_(self.host_data, non_blocking=True)
            self.target_buf.copy_(self.host_target, non_blocking=True)
            self.graph.replay()  # Replay captured graph (no launch overhead)

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.data_buf, self.target_buf, self.host_data, self.host_target, self.graph
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.graph is None:
            return "CUDA graph not initialized"
        if self.data_buf is None:
            return "Data buffer not initialized"
        if self.target_buf is None:
            return "Target buffer not initialized"
        # Check that graph can be replayed
        try:
            if self.data_buf.shape[0] != self.batch_size:
                return f"Data buffer batch size mismatch: expected {self.batch_size}, got {self.data_buf.shape[0]}"
            if self.target_buf.shape[0] != self.batch_size:
                return f"Target buffer batch size mismatch: expected {self.batch_size}, got {self.target_buf.shape[0]}"
        except Exception as e:
            return f"Buffer validation failed: {e}"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedPerformanceGraphsBenchmark(batch_size=128)


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5)
    )
    benchmark = OptimizedPerformanceGraphsBenchmark(batch_size=128)
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Optimized: Performance with CUDA Graphs")
    print("=" * 70)
    print(f"Batch size: {benchmark.batch_size}")
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

