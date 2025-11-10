"""HBM optimization: fused large-batch matmul that keeps data contiguous."""

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
from common.python.compile_utils import enable_tf32


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch3")
    return torch.device("cuda")


class OptimizedHbmBenchmark(Benchmark):
    """Processes the full tensor at once with torch.compile + FP16."""

    def __init__(self):
        self.device = resolve_device()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(22)
        enable_tf32()
        module = nn.Sequential(
            nn.Linear(1024, 2048, bias=False),
            nn.GELU(),
            nn.Linear(2048, 1024, bias=False),
        ).to(self.device).half()
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is not None:
            try:
                module = compile_fn(module, mode="reduce-overhead")
            except Exception:
                pass
        self.model = module.eval()
        self.inputs = torch.randn(4096, 1024, device=self.device, dtype=torch.float16).contiguous()
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.model is not None and self.inputs is not None
        with nvtx_range("optimized_hbm", enable=enable_nvtx):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _ = self.model(self.inputs)

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=25, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedHbmBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nOptimized HBM latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
