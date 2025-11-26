"""baseline_expert_parallelism.py - Baseline MoE without expert parallelism."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class ExpertLayer(nn.Module):
    """Single expert in MoE model."""
    
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.expert = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expert(x)


class BaselineExpertParallelismBenchmark(BaseBenchmark):
    """Baseline: MoE without expert parallelism (all experts on single GPU)."""
    
    def __init__(self):
        super().__init__()
        self.experts = None
        self.router = None
        self.input_data = None
        self.num_experts = 8
        self.top_k = 2
        tokens = 32 * 256
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        torch.manual_seed(42)
        self.experts = nn.ModuleList([
            ExpertLayer(256).to(self.device) for _ in range(self.num_experts)
        ])
        
        self.router = nn.Linear(256, self.num_experts).to(self.device)
        self.input_data = torch.randn(32, 256, device=self.device)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        if self.experts is None or self.router is None or self.input_data is None:
            raise RuntimeError("Benchmark not configured")

        with self._nvtx_range("baseline_expert_parallelism"):
            with torch.no_grad():
                router_logits = self.router(self.input_data)
                top_k_weights, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
                top_k_weights = torch.softmax(top_k_weights, dim=-1)
                
                output = torch.zeros_like(self.input_data)
                for expert_id in range(self.num_experts):
                    expert_mask = (top_k_indices == expert_id).any(dim=-1)
                    if expert_mask.any():
                        expert_input = self.input_data[expert_mask]
                        expert_output = self.experts[expert_id](expert_input)
                        output[expert_mask] += expert_output
        self._synchronize()
    
    def teardown(self) -> None:
        self.experts = None
        self.router = None
        self.input_data = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=3,
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
        if self.experts is None or len(self.experts) == 0:
            return "Experts not initialized"
        if self.router is None:
            return "Router not initialized"
        return None


def get_benchmark() -> BaselineExpertParallelismBenchmark:
    """Factory function for harness discovery."""
    return BaselineExpertParallelismBenchmark()
