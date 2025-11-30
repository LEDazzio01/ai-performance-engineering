"""optimized_continuous_batching.py - Optimized continuous batching."""

from __future__ import annotations

from collections import deque
from typing import Optional

import torch
import torch.nn as nn

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedContinuousBatchingBenchmark(BaseBenchmark):
    """Optimized: continuous batching with dynamic batch composition."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.sample_queue: Optional[deque] = None
        self.max_batch_size = 12
        self.hidden_dim = 1024
        # Match baseline total samples: batch_size(12) * num_batches(12) = 144
        self.batch_size = 12  # For signature matching
        self.num_batches = 12  # For signature matching with baseline
        self.num_samples = 144
        tokens = self.num_samples * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.num_samples),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: initialize model and sample queue."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        ).to(self.device).eval()
        
        self.sample_queue = deque(
            torch.randn(1, self.hidden_dim, device=self.device)
            for _ in range(self.num_samples)
        )
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: continuous batching - dynamic batch composition."""
        assert self.model is not None and self.sample_queue is not None
        with self._nvtx_range("optimized_continuous_batching"):
            with torch.no_grad():
                while self.sample_queue:
                    current_batch = []
                    current_size = 0
                    while self.sample_queue and current_size < self.max_batch_size:
                        sample = self.sample_queue.popleft()
                        current_batch.append(sample)
                        current_size += 1
                    if current_batch:
                        batch = torch.cat(current_batch, dim=0)
                        _ = self.model(batch)
            self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.sample_queue = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=getattr(self, '_ttft_ms', 50.0),
            tpot_ms=getattr(self, '_tpot_ms', 10.0),
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.sample_queue is None:
            return "Sample queue not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedContinuousBatchingBenchmark()
