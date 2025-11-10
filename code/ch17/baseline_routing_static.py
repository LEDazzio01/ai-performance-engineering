"""baseline_routing_static.py - Always route to the largest model."""

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
        raise RuntimeError("CUDA required for ch17")
    return torch.device("cuda")


class LargeModel(nn.Module):
    def __init__(self, hidden_dim: int = 2048, num_layers: int = 24):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.output = nn.Linear(hidden_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)


class BaselineRoutingStaticBenchmark(Benchmark):
    """Static routing baseline: every request uses the large model."""

    def __init__(self):
        self.device = resolve_device()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.batch_size = 16
        self.hidden_dim = 2048
        self.num_layers = 24

    def setup(self) -> None:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)

        self.model = LargeModel(self.hidden_dim, self.num_layers).to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.half()
        self.model.eval()

        dtype = next(self.model.parameters()).dtype
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=dtype)

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.model is not None and self.inputs is not None

        with nvtx_range("routing", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.model(self.inputs)

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.inputs is None:
            return "Model/input not initialized"
        return None


def get_benchmark() -> Benchmark:
    return BaselineRoutingStaticBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nBaseline Routing Static: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
