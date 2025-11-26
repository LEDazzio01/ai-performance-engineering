"""baseline_warp_specialization.py - Baseline without warp specialization."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from typing import Optional

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from ch10.workload_config import WORKLOAD


class BaselineWarpSpecializationPipelineBenchmark(BaseBenchmark):
    """Baseline: Sequential processing without warp specialization or pipelining."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.inputs_host = None
        self.workload = WORKLOAD
        self.micro_batches = self.workload.pipeline_micro_batches
        self.chunk_tokens = self.workload.pipeline_chunk_tokens
        self.hidden_dim = self.workload.pipeline_hidden_dim
        self._checksum = 0.0
        tokens = self.micro_batches * self.chunk_tokens
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.micro_batches),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model without warp specialization."""
        torch.manual_seed(42)
        
        # Baseline: Sequential model - no warp specialization
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        ).to(self.device).eval()
        
        # Match optimized workload size
        self.inputs_host = torch.randn(
            self.micro_batches,
            self.chunk_tokens,
            self.hidden_dim,
            pin_memory=True,
        )
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sequential forward pass."""
        with self._nvtx_range("baseline_warp_specialization_pipeline"):
            with torch.no_grad():
                # Baseline: Sequential processing
                # All warps do the same work - no specialization
                total = 0.0
                for idx in range(self.micro_batches):
                    host_chunk = self.inputs_host[idx]
                    device_chunk = host_chunk.to(self.device, non_blocking=False)
                    output = self.model(device_chunk)
                    total += float(output.sum().item())
                    self._synchronize()
                self._checksum = total
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.inputs_host = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )
}

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.inputs_host is None:
            return "Inputs not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineWarpSpecializationPipelineBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    config = benchmark.get_config()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Warp Specialization (Pipeline): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
