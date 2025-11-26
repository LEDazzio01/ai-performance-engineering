"""baseline_dataloader_default.py - Default DataLoader baseline (baseline)."""

from __future__ import annotations

from typing import Optional, Iterator

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class SyntheticDataset(Dataset):
    """Synthetic dataset for DataLoader benchmarking."""
    
    def __init__(self, num_samples: int = 1000, feature_dim: int = 1024):
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        base = torch.randn(num_samples, feature_dim)
        self.data = base
        self.labels = torch.randint(0, 10, (num_samples,))
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]
        enriched = torch.tanh(sample * 1.1) + torch.sin(sample * 0.5)
        normalized = enriched - enriched.mean()
        return normalized, self.labels[idx]


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


class BaselineDataloaderDefaultBenchmark(BaseBenchmark):
    """Default DataLoader baseline - no optimizations."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.dataloader = None
        self.optimizer = None
        self.criterion = None
        # Larger batches to increase input pipeline cost for the baseline.
        self.dataset_size = 1000
        self.batch_size = 64
        self.feature_dim = 1024
        self._data_iter: Optional[Iterator] = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.dataset_size // self.batch_size),
            tokens_per_iteration=float(self.dataset_size * self.feature_dim),
        )
    
    def setup(self) -> None:
        torch.manual_seed(42)
        
        self.model = SimpleModel(input_dim=self.feature_dim).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        
        dataset = SyntheticDataset(num_samples=self.dataset_size, feature_dim=self.feature_dim)
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            prefetch_factor=None,
        )
        
        self._data_iter = iter(self.dataloader)

        for _ in range(2):
            data, labels = self._next_batch()
            _ = self.model(data)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        if any(v is None for v in (self.model, self.optimizer, self.criterion, self._data_iter)):
            raise RuntimeError("Benchmark not configured")

        with self._nvtx_range("baseline_dataloader_default"):
            data, labels = self._next_batch()
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        self._synchronize()

    
    def teardown(self) -> None:
        self.model = None
        self.dataloader = None
        self.optimizer = None
        self.criterion = None
        self._data_iter = None
        super().teardown()
    
    def _next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.dataloader is not None
        if self._data_iter is None:
            self._data_iter = iter(self.dataloader)
        try:
            data, labels = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.dataloader)
            data, labels = next(self._data_iter)
        return data.to(self.device), labels.to(self.device)
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaselineDataloaderDefaultBenchmark:
    """Factory function for harness discovery."""
    return BaselineDataloaderDefaultBenchmark()
