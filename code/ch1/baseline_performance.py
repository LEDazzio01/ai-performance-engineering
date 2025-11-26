"""baseline_performance.py - Baseline performance benchmark (goodput measurement).

Implements BaseBenchmark for harness integration.
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

from common.python.compile_utils import compile_model  # Local helper applies TF32 + torch.compile defaults.

from typing import Optional

from common.python.benchmark_harness import (
    BaseBenchmark,
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


def _should_use_compile(device: torch.device) -> bool:
    """Decide whether to torch.compile the model.
    
    torch.compile is currently unstable on Blackwell (sm_100) for this
    benchmark and has been causing subprocess segfaults. Keep it disabled
    on CUDA to prefer eager for stability.
    """
    if device.type != "cuda":
        return False
    return False


class BaselinePerformanceBenchmark(BaseBenchmark):
    """Benchmark implementation following BaseBenchmark."""
    
    def __init__(self):
        super().__init__()
        self.device = resolve_device()
        self.model = None
        self.data = None
        self.target = None
        self.optimizer = None
        self.workload = WORKLOAD
        self.batch_size = self.workload.microbatch_size
        self.num_microbatches = self.workload.performance_microbatches
        self.microbatches = None
        self.targets = None
    
    def setup(self) -> None:
        """Setup: initialize model and data."""
        self.model = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
        )
        
        if _should_use_compile(self.device):
            self.model = compile_model(
                self.model.to(self.device),
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=False,
            )
        else:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        self.microbatches = [
            torch.randn(self.batch_size, 256, device=self.device)
            for _ in range(self.num_microbatches)
        ]
        self.targets = [
            torch.randint(0, 10, (self.batch_size,), device=self.device)
            for _ in range(self.num_microbatches)
        ]
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        # Warm up: run a few iterations so any torch.compile graph capture or
        # kernel autotuning occurs before the harness starts timing.
        for _ in range(3):
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(self.microbatches[0])
            loss = torch.nn.functional.cross_entropy(logits, self.targets[0])
            loss.backward()
            self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.optimizer.zero_grad(set_to_none=True)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_performance", enable=enable_nvtx):
            for data, target in zip(self.microbatches, self.targets):
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(data)
                loss = torch.nn.functional.cross_entropy(logits, target)
                loss.backward()
                self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.microbatches, self.targets, self.optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=5,
            warmup=1,
        )
    
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
        # Use the first microbatch as a validation probe so we avoid allocating
        # an additional tensor up front.
        probe = self.microbatches[0]
        try:
            with torch.no_grad():
                test_output = self.model(probe)
                if test_output.shape[0] != probe.shape[0]:
                    return f"Output shape mismatch: expected batch_size={probe.shape[0]}, got {test_output.shape[0]}"
                if test_output.shape[1] != 10:
                    return f"Output shape mismatch: expected num_classes=10, got {test_output.shape[1]}"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselinePerformanceBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5)
    )
    benchmark = BaselinePerformanceBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Baseline: Performance Basics")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
