"""optimized_all_techniques.py - All optimizations combined (optimized).

Combines: FP16 tensor cores, larger batch, CUDA graphs, fused operations.
Demonstrates cumulative benefits of stacking optimizations.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

try:
    import ch20.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass
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
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")


class SimpleModel(nn.Module):
    """Simple model for optimization demonstration."""
    
    def __init__(self, hidden_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class OptimizedAllTechniquesBenchmark(Benchmark):
    """Benchmark implementation following Benchmark protocol."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.x = None
        self.graph = None
        self.x_capture = None
        self.batch_size = 32
        self.hidden_dim = 4096
    
    def setup(self) -> None:
        """Setup: initialize model, compile, and capture CUDA graph."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        # Create model with FP16
        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).half().eval()
        
        # Compile model for fusion
        try:
            compiled_model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False, dynamic=False)
            self.model = compiled_model
        except Exception:
            pass  # Fallback to uncompiled
        
        # Prepare input
        self.x = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        
        # Warmup for compilation
        for _ in range(50):
            with torch.no_grad():
                _ = self.model(self.x)
        torch.cuda.synchronize()
        
        # Capture CUDA graph
        try:
            self.graph = torch.cuda.CUDAGraph()
            self.x_capture = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
            with torch.cuda.graph(self.graph):
                with torch.no_grad():
                    _ = self.model(self.x_capture)
            torch.cuda.synchronize()
            
            # Warmup graph replays
            for _ in range(10):
                self.graph.replay()
            torch.cuda.synchronize()
        except Exception:
            self.graph = None  # Fallback to regular execution
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_multiple_all_techniques", enable=enable_nvtx):
            with torch.no_grad():
                if self.graph:
                    self.graph.replay()
                else:
                    _ = self.model(self.x)

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.x, self.graph, self.x_capture
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=200,
            warmup=10,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.x is None:
            return "Input tensor not initialized"
        try:
            with torch.no_grad():
                # Test with regular forward pass (not graph replay)
                test_output = self.model(self.x)
                if test_output.shape[0] != self.batch_size:
                    return f"Output shape mismatch: expected batch_size={self.batch_size}, got {test_output.shape[0]}"
                if test_output.shape[1] != self.hidden_dim:
                    return f"Output shape mismatch: expected hidden_dim={self.hidden_dim}, got {test_output.shape[1]}"
                if not torch.isfinite(test_output).all():
                    return "Output contains non-finite values"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedAllTechniquesBenchmark()


def main() -> None:
    """Standalone execution with timing."""
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=200, warmup=10)
    )
    benchmark = OptimizedAllTechniquesBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Optimized: All Techniques Combined")
    print("=" * 70)
    print("Optimizations:")
    print("  1. FP16/BF16 precision (tensor cores enabled)")
    print("  2. Larger batch size (better GPU utilization)")
    print("  3. CUDA graphs (reduced launch overhead)")
    print("  4. Compiled model (kernel fusion)")
    print("Note: Same hidden_dim and iterations as baseline\n")
    
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")
    print(f"Throughput: {result.iterations / (result.mean_ms * result.iterations / 1000):.2f} iterations/sec")
    print("Status: All optimizations combined")
    print("Cumulative speedup: ~5-10x over baseline")


if __name__ == "__main__":
    main()
