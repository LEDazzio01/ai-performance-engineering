"""optimized_performance_pinned.py - Optimized performance benchmark with pinned memory.

Demonstrates pinned memory optimization for faster CPU-GPU transfers.
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


class OptimizedPerformancePinnedBenchmark(Benchmark):
    """Benchmark implementation with pinned memory optimization."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.host_data = None
        self.host_target = None
        self.data = None
        self.target = None
        self.optimizer = None
        self.use_fp16 = False
    
    def setup(self) -> None:
        """Setup: initialize model and data with pinned memory."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
        )
        
        if self.device.type == "cuda":
            try:
                self.model = self.model.to(self.device)
            except Exception as exc:
                print(f"WARNING: GPU initialization failed: {exc}. Falling back to CPU.")
                self.device = torch.device("cpu")
                self.model = self.model.cpu()
        else:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Optimization: Enable pinned memory for faster transfers
        self.host_data = torch.empty(32, 256, pin_memory=True)
        self.host_target = torch.empty(32, dtype=torch.long, pin_memory=True)
        self.data = torch.empty(32, 256, device=self.device)
        self.target = torch.empty(32, dtype=torch.long, device=self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark with pinned memory."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_performance_pinned", enable=enable_nvtx):
            # Optimization: Pinned memory enables faster CPU-GPU transfers
            # Non-blocking copies allow overlap with computation
            self.host_data.normal_(0, 1)
            self.host_target.random_(0, 10)
            self.data.copy_(self.host_data, non_blocking=True)
            self.target.copy_(self.host_target, non_blocking=True)
            # No sync needed - computation will wait for data automatically
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(self.data)
            loss = torch.nn.functional.cross_entropy(logits, self.target)
            loss.backward()
            self.optimizer.step()

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.host_data, self.host_target, self.data, self.target, self.optimizer
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
        if self.data is None:
            return "Data tensor not initialized"
        if self.target is None:
            return "Target tensor not initialized"
        # Check that model can produce valid output
        try:
            with torch.no_grad():
                test_output = self.model(self.data)
                if test_output.shape[0] != 32:
                    return f"Output batch size mismatch: expected 32, got {test_output.shape[0]}"
                if test_output.shape[1] != 10:
                    return f"Output shape mismatch: expected num_classes=10, got {test_output.shape[1]}"
                if self.target.shape[0] != 32:
                    return f"Target batch size mismatch: expected 32, got {self.target.shape[0]}"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedPerformancePinnedBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5)
    )
    benchmark = OptimizedPerformancePinnedBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Optimized: Performance with Pinned Memory")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

