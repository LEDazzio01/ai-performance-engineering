"""optimized_performance.py - Optimized performance benchmark with larger batch size."""

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

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from ch1.workload_config import WORKLOAD


class OptimizedPerformanceBatchBenchmark(BaseBenchmark):
    """Benchmark implementation with larger batch size optimization."""
    
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.workload = WORKLOAD
        self.batch_size = batch_size if batch_size != 32 else self.workload.microbatch_size
        self.model = None
        self.microbatches = None
        self.targets = None
        self.optimizer = None
        self.fusion = 4
        tokens = self.batch_size * self.workload.performance_microbatches * 256
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.workload.performance_microbatches),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: initialize model and data with larger batch."""
        
        from common.python.compile_utils import compile_model
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
        )
        
        if self.device.type == "cuda":
            # Optimization: Use FP16 for faster computation (tensor cores)
            self.model = self.model.half()
            dtype = torch.float16
            self.model = self.model.to(self.device)
            # Skip torch.compile for this small model - overhead exceeds benefit
            # The speedup comes from FP16 + batch fusion instead
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
        with self._nvtx_range("optimized_performance_batch"):
            # Optimization: Larger batch size improves GPU utilization
            # Process more samples per forward pass, reducing overhead per sample
            for data, target in zip(self._fused_batches, self._fused_targets):
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(data)
                loss = torch.nn.functional.cross_entropy(logits, target)
                loss.backward()
                self.optimizer.step()
        self._synchronize()

    
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
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_environment_metrics
        return compute_environment_metrics(
            gpu_count=getattr(self, 'gpu_count', 1),
            gpu_memory_gb=getattr(self, 'gpu_memory_gb', 80.0),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if not self.microbatches:
            return "Data not initialized"
        # Use first microbatch as validation probe
        probe = self.microbatches[0]
        try:
            with torch.no_grad():
                test_output = self.model(probe)
                if test_output.shape[0] != probe.shape[0]:
                    return f"Output batch size mismatch: expected {probe.shape[0]}, got {test_output.shape[0]}"
                if test_output.shape[1] != 10:
                    return f"Output shape mismatch: expected num_classes=10, got {test_output.shape[1]}"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        return None


def get_benchmark() -> BaseBenchmark:
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
