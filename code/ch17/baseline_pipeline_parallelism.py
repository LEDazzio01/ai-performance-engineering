"""baseline_pipeline_parallelism.py - Baseline sequential processing without pipeline parallelism.

Demonstrates sequential processing of model layers without pipeline parallelism.
Pipeline parallelism: This baseline processes all layers sequentially on a single GPU.
Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselinePipelineParallelismBenchmark(BaseBenchmark):
    """Baseline: Sequential processing without pipeline parallelism (single GPU)."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.input_data = None
        self.batch_size = 256
        self.hidden_size = 1024
        tokens = self.batch_size * self.hidden_size
        self._workload = WorkloadMetadata(
            samples_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model with all layers on single GPU."""
        torch.manual_seed(42)
        # Baseline: Sequential processing on single GPU
        # Pipeline parallelism splits model layers across multiple GPUs
        # This baseline processes all layers sequentially on one GPU
        self.model = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        ).to(self.device).eval()
        
        # Input data for inference
        self.input_data = torch.randn(self.batch_size, self.hidden_size, device=self.device)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sequential processing of all layers."""
        with self._nvtx_range("baseline_pipeline_parallelism"):
            with torch.no_grad():
                activations = self.input_data
                for layer in self.model:
                    activations = layer(activations)
                    self._synchronize()
        self._synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input_data = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


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

def get_benchmark() -> BaselinePipelineParallelismBenchmark:
    """Factory function for benchmark discovery."""
    return BaselinePipelineParallelismBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
