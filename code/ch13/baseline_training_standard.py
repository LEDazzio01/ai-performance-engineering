"""baseline_training_standard.py - Standard training without checkpointing (baseline). Stores all activations during forward pass - fast but memory-intensive.
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
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")


class DeepModel(nn.Module):
    """Deep model for demonstrating checkpoint benefits."""
    
    def __init__(self, hidden_dim=2048, num_layers=20):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        # Standard: Store all activations (memory-intensive)
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


class BaselineTrainingBenchmark(Benchmark):
    """Benchmark implementation following Benchmark protocol."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        
        # Larger model to better demonstrate memory savings with checkpointing
        # Increased layers: more activations stored, bigger memory benefit from checkpointing
        # Reduced to 40 layers to avoid GPU OOM (50 layers = 75GB params+gradients is too tight)
        # At 40 layers: ~60GB params+gradients, leaving safe margin for overhead
        self.batch_size = 8
        self.hidden_dim = 4096
        self.num_layers = 40  # Safe size that still shows checkpointing benefits
    
    def setup(self) -> None:
        """Setup: initialize model and data."""
        self.model = DeepModel(hidden_dim=self.hidden_dim, num_layers=self.num_layers)
        self.model = self.model.to(self.device).train()
        
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device)
        self.targets = torch.randn(self.batch_size, self.hidden_dim, device=self.device)
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_training_standard", enable=enable_nvtx):
            self.optimizer.zero_grad()
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward()
            self.optimizer.step()

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.inputs, self.targets, self.optimizer, self.criterion
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
        if self.inputs is None:
            return "Input tensor not initialized"
        if self.targets is None:
            return "Target tensor not initialized"
        
        try:
            with torch.no_grad():
                test_output = self.model(self.inputs)
                if test_output.shape != self.targets.shape:
                    return f"Output shape mismatch: expected {self.targets.shape}, got {test_output.shape}"
                if not torch.isfinite(test_output).all():
                    return "Output contains non-finite values"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineTrainingBenchmark()


def main() -> None:
    """Standalone execution with timing."""
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5)
    )
    benchmark = BaselineTrainingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Baseline: Standard Training (No Checkpointing)")
    print("=" * 70)
    print(f"Model: {benchmark.num_layers} layers, {benchmark.hidden_dim} hidden dim")
    print(f"Batch: {benchmark.batch_size}")
    print("Mode: Standard (stores all activations)\n")
    print(f"Average time per iteration: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")
    print("Status: Standard training (fast but memory-heavy)")
    print("\n Tip: Use gradient checkpointing for 30-50% memory reduction (10-30% slower)")


if __name__ == "__main__":
    main()
