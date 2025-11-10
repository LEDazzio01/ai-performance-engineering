"""Optimized single-GPU path that mimics DDP efficiency with torch.compile."""

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

from common.python.benchmark_harness import Benchmark, BenchmarkConfig
from common.python.compile_utils import enable_tf32


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch4")
    return torch.device("cuda:0")


class SimpleNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(x)))


class OptimizedDdpBenchmark(Benchmark):
    """Uses torch.compile + pinned prefetch for minimal CPU overhead."""

    def __init__(self):
        self.device = resolve_device()
        self.batch_size = 256
        self.input_size = 1024
        self.hidden_size = 256
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.inputs: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.batch_idx = 0

    def setup(self) -> None:
        enable_tf32()
        model = SimpleNet(self.input_size, self.hidden_size).to(self.device)
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is not None:
            try:
                model = compile_fn(model, mode="reduce-overhead")
            except Exception:
                pass
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        for _ in range(4):
            self.inputs.append(torch.randn(self.batch_size, self.input_size, device=self.device, dtype=torch.float16))
            self.targets.append(torch.randn(self.batch_size, 1, device=self.device, dtype=torch.float16))
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.model is not None and self.optimizer is not None
        idx = self.batch_idx % len(self.inputs)
        self.batch_idx += 1
        with nvtx_range("optimized_dataparallel", enable=enable_nvtx):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = self.model(self.inputs[idx])
                loss = nn.functional.mse_loss(output, self.targets[idx])
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    def teardown(self) -> None:
        self.model = None
        self.optimizer = None
        self.inputs = []
        self.targets = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedDdpBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nOptimized DDP latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
