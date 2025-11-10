"""optimized_dataloader_default.py - Tuned DataLoader optimization (optimized).

Optimized DataLoader with num_workers>0, pin_memory=True, and prefetch_factor.
Overlaps data loading with GPU computation for better GPU utilization.

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

from collections.abc import Iterator
from typing import Optional

from common.python.compile_utils import enable_tf32
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

class OptimizedDataloaderTunedBenchmark(Benchmark):
    """Optimized DataLoader - tuned for performance."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization

        # Optimization: Compile model for kernel fusion and optimization

        self.dataloader = None
        self.optimizer = None
        self.criterion = None
        self.dataset_size = 500
        self.batch_size = 32
        self.feature_dim = 1024
        self._data_iter: Optional[Iterator] = None
    
    def setup(self) -> None:
        """Setup: Initialize model and optimized DataLoader."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            enable_tf32()
        torch.manual_seed(42)
        
        # Compile model for better performance
        model = SimpleModel(input_dim=self.feature_dim).to(self.device)
        major, minor = torch.cuda.get_device_capability(self.device)
        compile_safe = major < 12  # CUDA 12.1 (Blackwell) hits torch.compile crashes
        if compile_safe:
            try:
                self.model = torch.compile(model, mode="reduce-overhead")
            except Exception:
                self.model = model
        else:
            self.model = model
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimized DataLoader with optimal settings
        # num_workers>0: Multi-threaded data loading (overlaps with GPU computation)
        # pin_memory=True: Pinned memory for faster CPU-GPU transfer
        # prefetch_factor: Prefetch batches ahead to hide I/O latency
        # persistent_workers: Keep workers alive to avoid startup overhead
        dataset = SyntheticDataset(num_samples=self.dataset_size, feature_dim=self.feature_dim)
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Disable shuffle for consistent benchmarking
            num_workers=4,  # Multi-threaded data loading
            pin_memory=True,  # Pinned memory for faster transfer
            prefetch_factor=4,  # Increased prefetch for better overlap
            persistent_workers=True,  # Keep workers alive
        )
        self._data_iter = iter(self.dataloader)
        
        # Warmup: run a couple of full steps so torch.compile + DataLoader overlap settle.
        for i, (data, labels) in enumerate(self.dataloader):
            if i >= 2:
                break
            # Data already on GPU (pin_memory + async transfer)
            data = data.to(self.device, non_blocking=True)  # Non-blocking transfer
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        torch.cuda.synchronize()
        self._data_iter = iter(self.dataloader)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - optimized DataLoader with overlapping."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("dataloader_default", enable=enable_nvtx):
            # Process one batch (optimized DataLoader: overlapped loading)
            if self._data_iter is None:
                self._data_iter = iter(self.dataloader)
            try:
                data, labels = next(self._data_iter)
            except StopIteration:
                self._data_iter = iter(self.dataloader)
                data, labels = next(self._data_iter)
            data = data.to(self.device, non_blocking=True)  # Non-blocking transfer
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def teardown(self) -> None:
        """Cleanup."""
        iterator = getattr(self, "_data_iter", None)
        if iterator is not None:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    pass
        del self.model, self.dataloader, self.optimizer, self.criterion
        self._data_iter = None
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
    return OptimizedDataloaderTunedBenchmark()

if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Tuned DataLoader: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
