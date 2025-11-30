"""optimized_multiple_unoptimized.py - AI-suggested optimizations applied.

Chapter 20: AI-Assisted Performance Optimizations

Optimizations applied (as AI would suggest):
1. BF16/FP16 for tensor core acceleration
2. Fused operations (single kernel instead of multiple)
3. Efficient normalization (single pass)
4. Single forward pass (no redundant computation)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedModel(nn.Module):
    """Model with AI-suggested optimizations applied."""
    
    def __init__(self, hidden_dim=2048):
        super().__init__()
        # Same architecture but will use BF16 + fused ops
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 4)
        self.fc3 = nn.Linear(hidden_dim * 4, hidden_dim)
    
    def forward(self, x):
        # Optimization: Use GELU which has fused implementations
        x = F.gelu(self.fc1(x))  # Fused linear+activation
        x = F.gelu(self.fc2(x))  # Fused linear+activation
        x = self.fc3(x)
        # Optimization: Use layer_norm instead of manual norm
        x = F.layer_norm(x, x.shape[-1:])
        return x


class OptimizedAllTechniquesBenchmark(BaseBenchmark):
    """Optimized: BF16 + fused ops + no redundant compute."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.x: Optional[torch.Tensor] = None
        self.batch_size = 128
        self.hidden_dim = 2048  # Match baseline
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        # Optimization 1: BF16 for tensor cores
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = OptimizedModel(hidden_dim=self.hidden_dim).to(self.device, dtype=dtype).eval()
        self.x = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=dtype)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(self.x)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.model is not None and self.x is not None
        with self._nvtx_range("multiple_techniques_optimized"):
            with torch.no_grad():
                # Optimization: Single forward pass (no redundant compute)
                out = self.model(self.x)
                _ = out.sum()  # Force materialization
            self._synchronize()
    
    def teardown(self) -> None:
        self.model = None
        self.x = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=200,
            warmup=20,
        )
    
    def get_workload_metadata(self):
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return optimization stack metrics."""
        return {
            "ch20.uses_bf16": 1.0,
            "ch20.uses_fused_ops": 1.0,
            "ch20.no_redundant_compute": 1.0,
        }

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.x is None:
            return "Input tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedAllTechniquesBenchmark()
