"""optimized_autograd_standard.py - Compiled autograd optimization (optimized).

Compiled autograd using torch.compile for faster backward pass.
Optimizes gradient computation through compilation.

Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

# Import arch_config to apply Triton patch for sm_12x support
# The patch removes 'a' suffix from sm_121a -> sm_121 for ptxas compatibility
try:
    import arch_config  # noqa: F401
except ImportError:
    pass  # Continue if arch_config not available
from typing import Optional

from common.python.compile_utils import (
    enable_tf32,
    is_torch_compile_supported_on_device,
)
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")


class SimpleModel(nn.Module):
    """Simple model for autograd comparison."""
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class OptimizedAutogradCompiledBenchmark(Benchmark):
    """Compiled autograd - uses torch.compile."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        self.batch_size = 32
        self.hidden_dim = 1024
        self._compiled = False
        self._logger = logging.getLogger(__name__)
    
    def setup(self) -> None:
        """Setup: Initialize compiled model and data."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            enable_tf32()
        torch.manual_seed(42)
        
        # Compile model for optimized autograd
        model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).half().train()
        support_ok, support_reason = is_torch_compile_supported_on_device(self.device.index)
        self._compiled = False

        if not support_ok and support_reason:
            self._logger.warning(
                "torch.compile is missing native SASS for this GPU (%s); continuing with PTX JIT fallback.",
                support_reason,
            )

        # Wrap torch.compile in try-except to handle compilation failures gracefully
        try:
            self.model = torch.compile(model, mode='reduce-overhead')  # Optimize backward pass
            self._compiled = True
            # Warmup to trigger compilation and catch errors early
            test_input = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
            test_target = torch.randn_like(test_input)
            for _ in range(2):
                self.model.zero_grad(set_to_none=True)
                output = self.model(test_input)
                loss = (output - test_target).pow(2).mean()
                loss.backward()
            torch.cuda.synchronize()
        except (RuntimeError, Exception) as e:
            # Fallback to eager mode if compilation fails
            error_msg = str(e)
            if (
                "generator" in error_msg.lower()
                or "SavedTensorHooks" in error_msg
                or "CppCompileError" in error_msg
            ):
                # Known PyTorch internal bugs - fall back to eager mode
                self._logger.warning("torch.compile failed (%s); falling back to eager mode.", error_msg.splitlines()[0])
                self.model = model
                self._compiled = False
            else:
                # Re-raise unknown errors
                raise

        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        self.targets = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        
        # Warmup (includes compilation) - full train steps to cover backward graph
        for _ in range(3):
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward()
            self.optimizer.step()
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - compiled autograd."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("autograd_standard", enable=enable_nvtx):
            self.optimizer.zero_grad()
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward()  # Compiled backward pass
            self.optimizer.step()

    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.inputs, self.targets, self.optimizer, self.criterion
        torch.cuda.empty_cache()
        if self._compiled:
            try:
                import torch._dynamo as _dynamo  # type: ignore
                _dynamo.reset()
            except Exception:
                pass
            self._compiled = False
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=180,  # torch.compile compilation can take 60-120 seconds
        )
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedAutogradCompiledBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Autograd Compiled: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
