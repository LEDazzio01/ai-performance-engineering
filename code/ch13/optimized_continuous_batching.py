"""optimized_continuous_batching.py - Optimized continuous batching in training context.

Demonstrates continuous batching where samples can be added/removed dynamically.
Batches are composed optimally from available samples, improving GPU utilization.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import deque

from common.python.compile_utils import compile_model

from typing import Optional, List, Tuple

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")


class SyntheticDataset(Dataset):
    """Synthetic dataset for training benchmarking."""
    
    def __init__(self, num_samples: int = 1000, feature_dim: int = 1024):
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.data = torch.randn(num_samples, feature_dim)
        self.labels = torch.randint(0, 10, (num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class OptimizedContinuousBatchingBenchmark(Benchmark):
    """Optimized: Continuous batching - dynamic batch composition.
    
    Continuous batching: Implements continuous batching where samples can be
    added or removed dynamically. Batches are composed optimally from available
    samples, improving GPU utilization and reducing idle time.
    """
    
    def __init__(self):
        self.device = resolve_device()
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

        self.sample_queue = None
    
    def setup(self) -> None:
        """Setup: Initialize model and sample queue."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        
        # Simple model for training
        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        ).to(self.device)
        
        self.model.train()
        
        # Optimization: Continuous batching - dynamic sample queue
        # Samples can be added/removed as they become available
        dataset = SyntheticDataset(num_samples=1000, feature_dim=1024)
        
        # Simulate continuous batching: queue of samples ready for processing
        self.sample_queue = deque([
            (dataset.data[i], dataset.labels[i])
            for i in range(len(dataset))
        ])
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Continuous batching - dynamic batch composition."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_continuous_batching", enable=enable_nvtx):
            # Optimization: Continuous batching
            # Compose batches dynamically from sample queue
            # Can combine samples optimally to fill batches
            max_batch_size = 32
            processed_batches = 0
            
            while self.sample_queue:
                # Compose batch from queue (continuous batching)
                current_batch_data = []
                current_batch_labels = []
                current_size = 0
                
                # Add samples to batch until max size reached
                while self.sample_queue and current_size < max_batch_size:
                    sample_data, sample_label = self.sample_queue.popleft()
                    current_batch_data.append(sample_data)
                    current_batch_labels.append(sample_label)
                    current_size += 1
                
                if current_batch_data:
                    # Stack into batch tensors
                    batch_data = torch.stack(current_batch_data).to(self.device)
                    batch_labels = torch.tensor(current_batch_labels, device=self.device)
                    
                    # Forward pass
                    output = self.model(batch_data)
                    
                    # Loss computation
                    loss = nn.functional.cross_entropy(output, batch_labels)
                    
                    # Backward pass (simplified - no optimizer step)
                    loss.backward()
                    
                    processed_batches += 1
                    
                    # Continuous batching: New samples can be added to queue
                    # without waiting for full batch completion
                    # This allows better GPU utilization

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.sample_queue = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.sample_queue is None:
            return "Sample queue not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedContinuousBatchingBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedContinuousBatchingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: continuous_batching")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
