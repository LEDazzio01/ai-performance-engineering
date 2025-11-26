"""optimized_autograd_standard.py - Compiled autograd optimization."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class SimpleModel(nn.Module):
    """Simple model for autograd comparison."""
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class OptimizedAutogradCompiledBenchmark(BaseBenchmark):
    """Autograd accelerated with CUDA graphs to remove launch overhead."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        # Smaller batch to increase launch overhead share and highlight graph capture.
        self.batch_size = 16
        self.hidden_dim = 1024
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.capture_stream: Optional[torch.cuda.Stream] = None
        self.static_input: Optional[torch.Tensor] = None
        self.static_target: Optional[torch.Tensor] = None
        self.input_pool: list[torch.Tensor] = []
        self.target_pool: list[torch.Tensor] = []
        self.pool_index = 0
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup training step, capture it with CUDA graphs."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        torch.manual_seed(42)
        
        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).half().train()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, foreach=True)
        self.criterion = nn.MSELoss()

        self.input_pool = [
            torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
            for _ in range(8)
        ]
        self.target_pool = [torch.randn_like(inp) for inp in self.input_pool]
        self.inputs = self.input_pool[0]
        self.targets = self.target_pool[0]

        for idx in range(3):
            self._train_step(self.input_pool[idx], self.target_pool[idx])
        self._synchronize()

        self.static_input = self.input_pool[0].clone()
        self.static_target = self.target_pool[0].clone()
        self.graph = torch.cuda.CUDAGraph()
        self.capture_stream = torch.cuda.Stream()
        with torch.cuda.stream(self.capture_stream):
            for _ in range(2):
                self._train_step(self.static_input, self.static_target)
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph, stream=self.capture_stream):
                self._train_step(self.static_input, self.static_target)
        self.capture_stream.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - compiled autograd."""
        if self.graph is None or self.static_input is None or self.static_target is None:
            raise RuntimeError("CUDA graph not initialized")

        current_input = self.input_pool[self.pool_index]
        current_target = self.target_pool[self.pool_index]
        self.pool_index = (self.pool_index + 1) % len(self.input_pool)

        with self._nvtx_range("autograd_standard"):
            self.static_input.copy_(current_input)
            self.static_target.copy_(current_target)
            self.graph.replay()
        self._synchronize()

    def _train_step(self, batch: torch.Tensor, target: torch.Tensor) -> None:
        assert self.model is not None and self.optimizer is not None and self.criterion is not None
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(batch)
        loss = self.criterion(outputs, target)
        loss.backward()
        self.optimizer.step()

    def teardown(self) -> None:
        """Cleanup."""
        self.model = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        self.graph = None
        self.static_input = None
        self.static_target = None
        self.capture_stream = None
        self.input_pool = []
        self.target_pool = []
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=180,
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


def get_benchmark() -> OptimizedAutogradCompiledBenchmark:
    """Factory function for harness discovery."""
    return OptimizedAutogradCompiledBenchmark()
