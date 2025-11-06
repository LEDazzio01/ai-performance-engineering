"""baseline_dataloader_default.py - Default DataLoader baseline (baseline).

Default DataLoader with no optimizations: num_workers=0, no pin_memory, no prefetch.
Results in CPU-GPU synchronization overhead and poor GPU utilization.

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


from typing import Optional

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


class SyntheticDataset(Dataset):
    """Synthetic dataset for DataLoader benchmarking."""
    
    def __init__(self, num_samples: int = 1000, feature_dim: int = 1024):
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        # Generate data upfront (simulating disk I/O)
        self.data = torch.randn(num_samples, feature_dim)
        self.labels = torch.randint(0, 10, (num_samples,))
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Simulate some processing overhead
        return self.data[idx], self.labels[idx]


class SimpleModel(nn.Module):
    """Simple model for training demonstration."""
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BaselineDataloaderDefaultBenchmark(Benchmark):
    """Default DataLoader baseline - no optimizations."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.dataloader = None
        self.optimizer = None
        self.criterion = None
        self.dataset_size = 500
        self.batch_size = 32
        self.feature_dim = 1024
    
    def setup(self) -> None:
        """Setup: Initialize model and default DataLoader."""
        torch.manual_seed(42)
        
        self.model = SimpleModel(input_dim=self.feature_dim).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        
        # Default DataLoader: num_workers=0, no pin_memory, no prefetch
        dataset = SyntheticDataset(num_samples=self.dataset_size, feature_dim=self.feature_dim)
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Disable shuffle for consistent benchmarking
            num_workers=0,  # Single-threaded (baseline)
            pin_memory=False,  # No pinned memory (baseline)
            prefetch_factor=None,  # No prefetching (baseline)
        )
        
        # Warmup
        for i, (data, labels) in enumerate(self.dataloader):
            if i >= 2:
                break
            data = data.to(self.device)
            labels = labels.to(self.device)
            _ = self.model(data)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - default DataLoader with no optimizations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_dataloader_default", enable=enable_nvtx):
            # Process one batch (default DataLoader: CPU-GPU sync overhead)
            data, labels = next(iter(self.dataloader))
            data = data.to(self.device)  # Synchronous transfer
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.dataloader, self.optimizer, self.criterion
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineDataloaderDefaultBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Default DataLoader: {result.mean_ms:.3f} ms")

