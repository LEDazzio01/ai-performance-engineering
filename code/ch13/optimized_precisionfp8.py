"""optimized_precisionfp8.py - AMP-based FP8 emulation benchmark."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import Benchmark, BenchmarkConfig


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13 precisionfp8")
    return torch.device("cuda")


class SimpleModel(nn.Module):
    """Two-layer MLP used for AMP + fake FP8 runs."""

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


def fake_fp8_cast(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize activations to float8 if supported, else fall back to fp16."""
    if hasattr(torch, "float8_e4m3fn"):
        return tensor.to(torch.float8_e4m3fn).to(torch.float16)
    return tensor.to(torch.float16)


class OptimizedFP8Benchmark(Benchmark):
    """Optimized FP8 path using PyTorch AMP + fake FP8 activations."""

    def __init__(self):
        self.device = resolve_device()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.scaler: Optional[GradScaler] = None
        self.batch_size = 256
        self.hidden_dim = 4096

    def setup(self) -> None:
        enable_tf32()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        torch.manual_seed(42)
        model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).train()
        model = model.half()
        self.model = model
        torch.manual_seed(42)
        self.inputs = torch.randn(
            self.batch_size,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.targets = torch.randn_like(self.inputs).to(torch.float16)
        self.inputs = self.inputs.to(torch.float32)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.scaler = GradScaler()

        for _ in range(5):
            self._train_step()
        torch.cuda.synchronize()
        self.optimizer.zero_grad(set_to_none=True)

    def _train_step(self) -> None:
        assert self.model and self.inputs is not None and self.targets is not None
        assert self.optimizer and self.criterion and self.scaler is not None
        self.optimizer.zero_grad(set_to_none=True)
        fp8_inputs = fake_fp8_cast(self.inputs)
        with autocast(device_type="cuda", dtype=torch.float16):
            outputs = self.model(fp8_inputs)
            loss = self.criterion(outputs, self.targets)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("optimized_precisionfp8", enable=enable_nvtx):
            self._train_step()

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        self.scaler = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=10)

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedFP8Benchmark()


if __name__ == "__main__":  # pragma: no cover
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    timing = result.timing.mean_ms if result.timing else 0.0
    print(f"\nOptimized Precision FP8 (AMP-based, precisionfp8 pair): {timing:.3f} ms")
