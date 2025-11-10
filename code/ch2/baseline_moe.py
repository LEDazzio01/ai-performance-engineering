"""Dense MoE baseline for Chapter 2."""

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


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch2 MoE example")
    return torch.device("cuda")


class DenseExpert(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class DenseMoELayer(nn.Module):
    """Naive implementation that evaluates every expert for every token."""

    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.experts = nn.ModuleList(DenseExpert(hidden_dim) for _ in range(num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        stacked = torch.stack(expert_outputs, dim=0)
        return stacked.mean(dim=0)


class BaselineMoEBenchmark(Benchmark):
    """Runs a dense MoE forward pass with no routing."""

    def __init__(self):
        self.device = resolve_device()
        self.hidden_dim = 768
        self.num_experts = 16
        self.batch = 16
        self.seq = 256
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(0)
        model = DenseMoELayer(self.hidden_dim, self.num_experts).to(self.device).float().eval()
        self.model = model
        self.inputs = torch.randn(self.batch, self.seq, self.hidden_dim, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.model is not None and self.inputs is not None
        with nvtx_range("baseline_moe_dense", enable=enable_nvtx):
            with torch.no_grad():
                outputs = []
                for expert in self.model.experts:
                    outputs.append(expert(self.inputs))
                    torch.cuda.synchronize()
                _ = torch.stack(outputs, dim=0).mean(dim=0)

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=2)

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
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nBaseline dense MoE latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
