"""baseline_continuous_batching.py - Baseline static batching in training context.

Demonstrates static batching where batches are processed sequentially.
No continuous batching: batches wait for full completion before next batch starts.
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

from common.python.compile_utils import compile_model

from typing import Optional

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


class BaselineContinuousBatchingBenchmark(Benchmark):
    """Baseline: Static batching - batches processed sequentially.
    
    Continuous batching: This baseline does not implement continuous batching.
    Batches are processed one at a time, with full synchronization between batches.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.dataloader = None
    
    def setup(self) -> None:
        """Setup: Initialize model and static dataloader."""
        torch.manual_seed(42)
        
        # Simple model for training
        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        ).to(self.device)
        
        self.model.train()
        
        # Baseline: Static batching - fixed batch sizes, processed sequentially
        # Continuous batching allows dynamic batch composition
        dataset = SyntheticDataset(num_samples=1000, feature_dim=1024)
        self.dataloader = DataLoader(
            dataset,
            batch_size=32,  # Fixed batch size
            shuffle=False,
            num_workers=0,  # No multiprocessing
            pin_memory=False,
        )
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Static batching - sequential processing."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_continuous_batching", enable=enable_nvtx):
            # Baseline: Process batches sequentially
            # Each batch must complete fully before next batch starts
            # No continuous batching: cannot add/remove samples mid-batch
            for batch_data, batch_labels in self.dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                output = self.model(batch_data)
                
                # Loss computation (simplified)
                loss = nn.functional.cross_entropy(output, batch_labels)
                
                # Backward pass (simplified - no optimizer step)
                loss.backward()
                
                # Wait for completion before next batch
                torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.dataloader = None
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
        if self.dataloader is None:
            return "DataLoader not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineContinuousBatchingBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineContinuousBatchingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: continuous_batching")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
