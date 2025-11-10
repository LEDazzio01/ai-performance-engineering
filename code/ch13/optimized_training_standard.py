"""optimized_training_standard.py - Gradient checkpointing optimization.

Recomputes activations during backward - slower but memory-efficient.
Enables training larger models that wouldn't fit otherwise.
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
from torch.utils.checkpoint import checkpoint

from typing import Optional

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode
)
from ch13.workload_config import WORKLOAD

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")

class DeepModel(nn.Module):
    """Deep model with gradient checkpointing."""
    
    def __init__(self, hidden_dim=2048, num_layers=20, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                # Gradient checkpointing: Recompute activations during backward
                x = checkpoint(lambda x: torch.relu(layer(x)), x, use_reentrant=False)
            else:
                x = torch.relu(layer(x))
        return x

class OptimizedCheckpointBenchmark(Benchmark):
    """Benchmark implementation following Benchmark protocol."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization

        # Optimization: Compile model for kernel fusion and optimization

        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        self.workload = WORKLOAD
        self.hidden_dim = self.workload.training_hidden_dim
        self.num_layers = self.workload.training_layers_optimized
        self.global_batch = self.workload.global_batch_size
        self.micro_batch = self.workload.micro_batch_size
        self.accum_steps = self.global_batch // self.micro_batch
        self.batch_size = self.micro_batch
    
    def setup(self) -> None:
        """Setup: initialize model and data."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            enable_tf32()
        self.model = DeepModel(hidden_dim=self.hidden_dim, num_layers=self.num_layers, use_checkpoint=True)
        self.model = self.model.to(self.device)
        # Keep model in FP32 - checkpointing is about memory, not precision
        # Converting to FP16 would require matching input dtype
        self.model.train()
        self.inputs = torch.randn(self.global_batch, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.targets = torch.randn(self.global_batch, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("training_standard", enable=enable_nvtx):
            self.optimizer.zero_grad(set_to_none=True)
            for start in range(0, self.global_batch, self.micro_batch):
                end = start + self.micro_batch
                inputs = self.inputs[start:end]
                targets = self.targets[start:end]
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets) / self.accum_steps
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
    return OptimizedCheckpointBenchmark()

def main() -> None:
    """Standalone execution with timing."""
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5)
    )
    benchmark = OptimizedCheckpointBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Optimized: Gradient Checkpointing")
    print("=" * 70)
    print(f"Model: {benchmark.num_layers} layers, {benchmark.hidden_dim} hidden dim")
    print(f"Batch: {benchmark.batch_size}")
    print("Mode: Checkpointing (recomputes activations)")
    print("Note: Same workload size as baseline\n")
    
    print(f"Average time per iteration: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")
    print("Status: Checkpointing (30-50% memory reduction, 10-30% slower)")
    print("Benefit: Enables training larger models")

if __name__ == "__main__":
    main()
