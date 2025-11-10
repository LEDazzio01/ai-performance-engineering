"""baseline_moe.py - Dense MoE routing baseline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from common.python.benchmark_harness import Benchmark, BenchmarkConfig
from ch16.moe_workload import (
    resolve_moe_workload,
    MOE_HIDDEN_DIM,
    MOE_NUM_EXPERTS,
)


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch16")
    return torch.device("cuda")


class DenseMoELayer(nn.Module):
    """Naive MoE implementation that runs every expert for every token."""

    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for expert in self.experts:
            outputs.append(expert(x))
        return torch.stack(outputs, dim=0).mean(dim=0)


class BaselineMoEBenchmark(Benchmark):
    """Dense MoE baseline: all experts fire every iteration."""

    def __init__(self):
        self.device = resolve_device()
        workload = resolve_moe_workload()
        self.batch_size = workload.batch_size
        self.seq_len = workload.seq_len
        self.hidden_dim = MOE_HIDDEN_DIM
        self.num_experts = MOE_NUM_EXPERTS
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.backends.cudnn.benchmark = True
        self.model = DenseMoELayer(self.hidden_dim, self.num_experts).to(self.device).half().eval()
        torch.manual_seed(42)
        self.inputs = torch.randn(
            self.batch_size,
            self.seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float16,
        )
        # Warm up once so that CUDA graph state is amortized.
        with torch.no_grad():
            _ = self.model(self.inputs)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.model is not None and self.inputs is not None
        with nvtx_range("baseline_moe_dense", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.model(self.inputs)

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=5,
            warmup=2,
            setup_timeout_seconds=180,
            measurement_timeout_seconds=180,
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.inputs is None:
            return "Model/input not initialized"
        return None


def get_benchmark() -> Benchmark:
    return BaselineMoEBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5),
    )
    result = harness.benchmark(get_benchmark())
    timing = result.timing.mean_ms if result.timing else 0.0
    print(f"\nBaseline Dense MoE: {timing:.3f} ms")
