"""Optimized single-GPU training - avoids DataParallel overhead.

Chapter 4: Parallelization Strategies on a Single Node

This optimized version demonstrates efficient single-GPU training:
- Data pre-staged on GPU (no CPU-GPU copies each iteration)
- BF16 autocast for tensor core acceleration
- Efficient optimizer with set_to_none=True
- No DataParallel wrapper overhead

The baseline uses DataParallel which is an anti-pattern on single GPU:
- Forces CPU-GPU data copies even when unnecessary
- GIL contention on forward/backward sync
- Extra Python overhead for device coordination
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
import torch.optim as optim

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from core.utils.compile_utils import enable_tf32


class SimpleNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # Same architecture as baseline for fair comparison
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        return self.linear3(x)


class OptimizedDdpBenchmark(BaseBenchmark):
    """Optimized: Direct GPU execution without DataParallel wrapper.
    
    Key optimizations vs DataParallel baseline:
    1. Data pre-staged on GPU (no CPU->GPU copies per iteration)
    2. No DataParallel wrapper overhead
    3. BF16 mixed precision for tensor cores
    4. Efficient zero_grad with set_to_none=True
    """

    def __init__(self):
        super().__init__()
        self.batch_size = 512
        self.input_size = 1024
        self.hidden_size = 256
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.inputs: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.batch_idx = 0
        tokens = self.batch_size * self.input_size
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        enable_tf32()
        torch.manual_seed(42)
        
        # Direct model on GPU - no DataParallel wrapper
        self.model = SimpleNet(self.input_size, self.hidden_size).to(
            self.device, dtype=torch.bfloat16
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        # Pre-stage data on GPU to avoid CPU-GPU copies
        for _ in range(4):
            self.inputs.append(torch.randn(
                self.batch_size, self.input_size, 
                device=self.device, dtype=torch.bfloat16
            ))
            self.targets.append(torch.randn(
                self.batch_size, 1, 
                device=self.device, dtype=torch.bfloat16
            ))
        
        # Warmup
        for _ in range(5):
            idx = 0
            output = self.model(self.inputs[idx])
            loss = nn.functional.mse_loss(output, self.targets[idx])
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.model is not None and self.optimizer is not None
        idx = self.batch_idx % len(self.inputs)
        self.batch_idx += 1
        
        with nvtx_range("optimized_dataparallel", enable=enable_nvtx):
            # Direct forward/backward - no DataParallel overhead
            output = self.model(self.inputs[idx])
            loss = nn.functional.mse_loss(output, self.targets[idx])
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        self._synchronize()

    def teardown(self) -> None:
        self.model = None
        self.optimizer = None
        self.inputs = []
        self.targets = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedDdpBenchmark()


if __name__ == "__main__":
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=10),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nOptimized DDP latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
