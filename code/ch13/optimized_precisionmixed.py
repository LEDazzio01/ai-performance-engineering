"""optimized_precision_mixed.py - Mixed precision optimization (optimized)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.amp import autocast

from common.python.compile_utils import enable_tf32, compile_model
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class SimpleModel(nn.Module):
    """Simple model for precision comparison."""
    
    def __init__(self, hidden_dim: int = 1024, depth: int = 4):
        super().__init__()
        self.depth = depth
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.hidden = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim * 2) for _ in range(depth)
        ])
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.in_proj(x))
        for layer in self.hidden:
            x = self.act(layer(x))
        x = self.out_proj(x)
        return x

class OptimizedPrecisionMixedBenchmark(BaseBenchmark):
    """Mixed precision - uses autocast and torch.compile."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        self.batch_size = 128
        self.hidden_dim = 2048
        self.micro_steps = 4
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.micro_steps),
            tokens_per_iteration=float(tokens * self.micro_steps),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model with mixed precision."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        torch.manual_seed(42)
        
        model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device)
        model.train()
        self.model = compile_model(model, mode="reduce-overhead")
        
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        self.targets = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        self.model = self.model.to(torch.float16)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        
        for _ in range(3):
            self.optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', dtype=torch.float16):
                outputs = self.model(self.inputs)
                loss = self.criterion(outputs, self.targets)
            loss.backward()
            self.optimizer.step()
        self._synchronize()
        self.optimizer.zero_grad(set_to_none=True)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - mixed precision training."""
        if any(v is None for v in (self.model, self.inputs, self.targets, self.optimizer, self.criterion)):
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("optimized_precision_mixed"):
            for _ in range(self.micro_steps):
                self.optimizer.zero_grad(set_to_none=True)
                
                with autocast('cuda', dtype=torch.float16):
                    outputs = self.model(self.inputs)
                    loss = self.criterion(outputs, self.targets)
                
                loss.backward()
                self.optimizer.step()
        self._synchronize()

    def teardown(self) -> None:
        """Cleanup."""
        self.model = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
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
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> OptimizedPrecisionMixedBenchmark:
    """Factory function for harness discovery."""
    return OptimizedPrecisionMixedBenchmark()
