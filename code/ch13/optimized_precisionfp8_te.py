"""optimized_precisionfp8_te.py - Transformer Engine FP8 benchmark."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
from torch.optim import Optimizer

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import Benchmark, BenchmarkConfig


def _preload_torch_cuda_symbols() -> None:
    """Ensure torch CUDA shared objects are loaded with RTLD_GLOBAL."""
    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    libs = [
        "libtorch_cuda.so",
        "libtorch_cuda_linalg.so",
        "libtorch_nvshmem.so",
        "libc10_cuda.so",
    ]
    for name in libs:
        candidate = torch_lib_dir / name
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)


_preload_torch_cuda_symbols()

try:
    from transformer_engine.pytorch import Linear as TELinear, fp8_autocast
    from transformer_engine.common import recipe as te_recipe

    TE_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    TE_AVAILABLE = False
    TE_IMPORT_ERROR = exc


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13 precisionfp8")
    return torch.device("cuda")


class TEFP8MLP(nn.Module):
    """Two-layer MLP built with Transformer Engine Linear layers."""

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = TELinear(hidden_dim, hidden_dim * 2, bias=True)
        self.fc2 = TELinear(hidden_dim * 2, hidden_dim, bias=True)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class OptimizedTEFP8Benchmark(Benchmark):
    """Optimized FP8 path using Transformer Engine."""

    def __init__(self):
        if not TE_AVAILABLE:
            raise RuntimeError(
                "Transformer Engine is required for optimized_precisionfp8_te. "
                f"(import error: {TE_IMPORT_ERROR})"
            )
        self.device = resolve_device()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.optimizer: Optional[Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.fp8_recipe = te_recipe.DelayedScaling(
            margin=0,
            interval=1,
            amax_history_len=16,
        )
        self.batch_size = 256
        self.hidden_dim = 4096

    def setup(self) -> None:
        enable_tf32()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        model = TEFP8MLP(hidden_dim=self.hidden_dim).to(self.device).train()
        self.model = model
        torch.manual_seed(42)
        self.inputs = torch.randn(
            self.batch_size,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.targets = torch.randn_like(self.inputs)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

        for _ in range(5):
            self._train_step()
        torch.cuda.synchronize()
        self.optimizer.zero_grad(set_to_none=True)

    def _train_step(self) -> None:
        assert self.model and self.inputs is not None and self.targets is not None
        assert self.optimizer and self.criterion
        self.optimizer.zero_grad(set_to_none=True)
        with fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
        loss.backward()
        self.optimizer.step()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("optimized_precisionfp8_te", enable=enable_nvtx):
            self._train_step()

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=10)

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedTEFP8Benchmark()


if __name__ == "__main__":  # pragma: no cover
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    timing = result.timing.mean_ms if result.timing else 0.0
    print(f"\nOptimized Precision FP8 (Transformer Engine): {timing:.3f} ms")
