"""Docker baseline: host batches copied synchronously each iteration.

This benchmark demonstrates inefficient data loading - using non-pinned memory
and blocking H2D copies. The optimized version uses pinned memory and prefetching.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import os
from core.benchmark.smoke import is_smoke_mode

from core.optimization.allocator_tuning import log_allocator_guidance
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class BaselineDockerBenchmark(BaseBenchmark):
    """Simulates a non-containerized setup with blocking H2D copies."""

    def __init__(self):
        super().__init__()
        low_mem = is_smoke_mode()
        # Match optimized workload for fair comparison
        self.input_dim = 2048 if low_mem else 4096
        self.hidden_dim = 2048 if low_mem else 4096
        self.output_dim = 1024 if low_mem else 2048
        self.batch_size = 512 if low_mem else 1024  # Large batch = significant H2D
        self.num_batches = 4 if low_mem else 8
        self.jitter_exemption_reason = "Docker benchmark: fixed batch size for comparison"
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.host_batches: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.batch_idx = 0
        self.output: Optional[torch.Tensor] = None
        # Training benchmarks don't support jitter check - outputs change due to weight updates
        self.jitter_exemption_reason = "Training benchmark: outputs change each iteration due to gradient updates"
        # Register workload metadata in __init__ for compliance checks
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(self.batch_size * self.input_dim * 4),  # float32
        )

    def setup(self) -> None:
        torch.manual_seed(101)
        log_allocator_guidance("ch03/baseline_docker", optimized=False)
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        ).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2)

        for _ in range(self.num_batches):
            self.host_batches.append(torch.randn(self.batch_size, self.input_dim, dtype=torch.float32))
            self.targets.append(torch.randn(self.batch_size, self.output_dim, dtype=torch.float32))
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.model is not None and self.optimizer is not None

        idx = self.batch_idx % len(self.host_batches)
        host_x = self.host_batches[idx]
        host_y = self.targets[idx]
        self.batch_idx += 1

        with nvtx_range("baseline_docker", enable=enable_nvtx):
            x = self.to_device(host_x)  # blocking copy (tensor not pinned)
            y = self.to_device(host_y)
            out = self.model(x)
            loss = torch.nn.functional.mse_loss(out, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
        
        # Store output for verification
        self.output = out.detach()

    def teardown(self) -> None:
        self.model = None
        self.optimizer = None
        self.host_batches = []
        self.targets = []
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        low_mem = is_smoke_mode()
        # Minimum warmup=5 even in smoke mode to exclude JIT overhead
        return BenchmarkConfig(iterations=5 if low_mem else 20, warmup=5 if low_mem else 10)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_system_config_metrics
        return compute_system_config_metrics(
            numa_nodes=getattr(self, 'numa_nodes', 1),
            cpu_cores=getattr(self, 'cpu_cores', 64),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is not None:
            return self.output.detach().clone()
        return torch.tensor([0.0], dtype=torch.float32, device=self.device)
    
    def get_output_tolerance(self) -> tuple:
        """Return custom tolerance for training output comparison.
        
        Training with SGD has some non-determinism due to CUDA operations.
        """
        return (1e-3, 1e-3)

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"batch_size": self.batch_size, "num_batches": self.num_batches}


def get_benchmark() -> BaseBenchmark:
    return BaselineDockerBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
