"""baseline_continuous_batching.py - Baseline static batching in disaggregated inference context."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineContinuousBatchingBenchmark(BaseBenchmark):
    """Baseline: static batching, sequential processing."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.batches: Optional[list[torch.Tensor]] = None
        self.batch_size = 12
        self.hidden_dim = 1024
        self.num_batches = 12
        tokens = self.batch_size * self.hidden_dim * self.num_batches
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.num_batches),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: initialize model and static batches."""
        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        ).to(self.device).eval()
        
        self.batches = [
            torch.randn(self.batch_size, self.hidden_dim, device=self.device)
            for _ in range(self.num_batches)
        ]
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: static batches processed sequentially."""
        assert self.model is not None and self.batches is not None
        with self._nvtx_range("baseline_continuous_batching"):
            with torch.no_grad():
                for batch in self.batches:
                    _ = self.model(batch)
            self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.batches = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_inference_metrics
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
        if self.batches is None:
            return "Batches not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineContinuousBatchingBenchmark()
