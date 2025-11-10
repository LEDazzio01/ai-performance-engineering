"""optimized_performance.py - Optimized performance benchmark with larger batch size.

Demonstrates how larger batch sizes improve GEMM efficiency.
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


from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from ch1.workload_config import WORKLOAD


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


class OptimizedPerformanceBatchBenchmark(Benchmark):
    """Benchmark implementation with larger batch size optimization."""
    
    def __init__(self, batch_size: int = 32):
        self.device = resolve_device()
        self.workload = WORKLOAD
        self.batch_size = batch_size if batch_size != 32 else self.workload.microbatch_size
        self.model = None
        self.microbatches = None
        self.targets = None
        self.optimizer = None
        self.fusion = 4
    
    def setup(self) -> None:
        """Setup: initialize model and data with larger batch."""
        
        try:
            from common.python.compile_utils import compile_model
        except ImportError:
            compile_model = lambda m, **kwargs: m
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
        )
        
        if self.device.type == "cuda":
            try:
                # Optimization: Use FP16 for faster computation - this is the key optimization
                # FP16 provides 2x theoretical speedup on modern GPUs
                # Convert to FP16 before moving to device for efficiency
                try:
                    self.model = self.model.half()
                    dtype = torch.float16
                except Exception:
                    dtype = torch.float32
                
                self.model = self.model.to(self.device)
                # Optimization: Compile model for kernel fusion (same as baseline)
                # FP16 + compilation provides best performance
                self.model = compile_model(self.model, mode="reduce-overhead", fullgraph=False, dynamic=False)
            except Exception as exc:
                print(f"WARNING: GPU initialization failed: {exc}. Falling back to CPU.")
                self.device = torch.device("cpu")
                self.model = self.model.cpu()
                dtype = torch.float32
        else:
            self.model = self.model.to(self.device)
            dtype = torch.float32
        
        # Match baseline: use eval() mode (baseline has this even though it does backward pass)
        self.model.eval()
        torch.manual_seed(42)
        microbatches = [
            torch.randn(self.batch_size, 256, device=self.device, dtype=dtype).contiguous()
            for _ in range(self.workload.performance_microbatches)
        ]
        targets = [
            torch.randint(0, 10, (self.batch_size,), device=self.device)
            for _ in range(self.workload.performance_microbatches)
        ]
        self.microbatches = microbatches
        self.targets = targets
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        # Warm up compiled model so the measurement loop only sees steady-state cost.
        for _ in range(3):
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(self.microbatches[0])
            loss = torch.nn.functional.cross_entropy(logits, self.targets[0])
            loss.backward()
            self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.optimizer.zero_grad(set_to_none=True)
        # Pre-build fused batches so the benchmark loop can issue fewer, larger kernels.
        self._fused_batches = []
        self._fused_targets = []
        for start in range(0, len(self.microbatches), self.fusion):
            batch = torch.cat(self.microbatches[start : start + self.fusion], dim=0)
            target = torch.cat(self.targets[start : start + self.fusion], dim=0)
            self._fused_batches.append(batch)
            self._fused_targets.append(target)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_performance_batch", enable=enable_nvtx):
            # Optimization: Larger batch size improves GPU utilization
            # Process more samples per forward pass, reducing overhead per sample
            # This is the key optimization - larger batches are more efficient
            for data, target in zip(self._fused_batches, self._fused_targets):
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(data)
                loss = torch.nn.functional.cross_entropy(logits, target)
                loss.backward()
                self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            
            # Optimization: Larger batch benefits
            # - Better GPU utilization (more parallelism)
            # - Reduced overhead per sample
            # - Higher throughput

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.microbatches, self.targets, self.optimizer
        self._fused_batches = None
        self._fused_targets = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=5,
            warmup=1,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.microbatches is None:
            return "Data not initialized"
        # Check that model can produce valid output
        try:
            with torch.no_grad():
                test_output = self.model(self.data)
                if test_output.shape[0] != self.batch_size:
                    return f"Output batch size mismatch: expected {self.batch_size}, got {test_output.shape[0]}"
                if test_output.shape[1] != 10:
                    return f"Output shape mismatch: expected num_classes=10, got {test_output.shape[1]}"
                if self.target.shape[0] != self.batch_size:
                    return f"Target batch size mismatch: expected {self.batch_size}, got {self.target.shape[0]}"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedPerformanceBatchBenchmark(batch_size=32)


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5)
    )
    benchmark = OptimizedPerformanceBatchBenchmark(batch_size=32)
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Optimized: Performance with Larger Batch Size")
    print("=" * 70)
    print(f"Batch size: {benchmark.batch_size}")
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
