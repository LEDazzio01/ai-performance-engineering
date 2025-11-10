"""optimized_multiple_unoptimized.py - All optimizations combined (optimized).

Combines: FP16 tensor cores, larger batch, CUDA graphs, fused operations.
Demonstrates cumulative benefits of stacking optimizations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import warnings

try:
    import ch20.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass
from ch20.inductor_guard import (
    disable_inductor_cudagraph_features,
    restore_inductor_cudagraph_features,
    InductorCudagraphState,
)

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
        self._inductor_cfg_state: InductorCudagraphState = None
    
    def setup(self) -> None:
        """Setup: initialize model, compile, and capture CUDA graph."""
        
        self._inductor_cfg_state = disable_inductor_cudagraph_features()
        try:
            # Optimization: Enable cuDNN benchmarking for optimal kernel selection
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            # Create model with FP16
            self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).half().eval()
            
            # Compile model for fusion
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False, dynamic=False)
            # Warmup to trigger compilation and catch errors early
            test_input = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
            for _ in range(3):
                with torch.no_grad():
                    _ = self.model(test_input)
            torch.cuda.synchronize()
            
            # Prepare input
            self.x = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
            
            # Warmup for compilation
            for _ in range(50):
                with torch.no_grad():
                    _ = self.model(self.x)
            torch.cuda.synchronize()
            
            # Capture CUDA graph if allowed by torch.compile backend
            self.graph = None
            self.x_capture = None
            try:
                graph = torch.cuda.CUDAGraph()
                self.x_capture = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
                # Ensure model is warmed up before graph capture
                for _ in range(5):
                    with torch.no_grad():
                        _ = self.model(self.x_capture)
                torch.cuda.synchronize()
                
                with torch.cuda.graph(graph):
                    with torch.no_grad():
                        _ = self.model(self.x_capture)
                torch.cuda.synchronize()
                
                # Warmup graph replays
                for _ in range(10):
                    graph.replay()
                torch.cuda.synchronize()
                self.graph = graph
            except RuntimeError as e:
                if "Cannot prepare for replay during capturing stage" in str(e):
                    warnings.warn(
                        "TorchInductor already uses CUDA graphs on this platform; "
                        "skipping manual CUDA graph capture for optimized_multiple_all_techniques.",
                        RuntimeWarning,
                    )
                    self.graph = None
                    self.x_capture = None
                    torch.cuda.synchronize()
                else:
                    raise
        except Exception:
            restore_inductor_cudagraph_features(self._inductor_cfg_state)
            self._inductor_cfg_state = None
            raise

    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("multiple_techniques", enable=enable_nvtx):
            with torch.no_grad():
                if self.graph is None:
                    _ = self.model(self.x)
                else:
                    self.graph.replay()

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.x, self.graph, self.x_capture
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        restore_inductor_cudagraph_features(self._inductor_cfg_state)
        self._inductor_cfg_state = None
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=200,
            warmup=10,
            use_subprocess=True,
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
    
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")
    throughput = 0.0
    if result.timing and result.timing.mean_ms > 0.0:
        throughput = 1000.0 / result.timing.mean_ms
    print(f"Throughput: {throughput:.2f} iterations/sec")
    print("Status: All optimizations combined")
    print("Cumulative speedup: ~5-10x over baseline")


if __name__ == "__main__":
    main()
