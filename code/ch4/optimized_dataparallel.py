"""Optimized single-GPU path that mimics DDP efficiency with torch.compile."""

from __future__ import annotations

import os
from common.python.smoke import is_smoke_mode
import sys
from pathlib import Path
from typing import Optional, List

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.optim as optim

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from common.python.compile_utils import enable_tf32


class SimpleNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(x)))


class OptimizedDdpBenchmark(BaseBenchmark):
    """Uses torch.compile + pinned prefetch for minimal CPU overhead."""

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
        self._compiled = False
        self._compile_error: Optional[str] = None
        tokens = self.batch_size * self.input_size
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        enable_tf32()
        base_model = SimpleNet(self.input_size, self.hidden_size).to(self.device)
        compile_fn = getattr(torch, "compile", None)
        low_mem = is_smoke_mode()

        # Torch compile can be heavy in constrained environments; skip in low-memory
        # runs and fall back to eager if compilation fails.
        if compile_fn is not None and not low_mem:
            try:
                compiled = compile_fn(base_model, mode="reduce-overhead")
                self.model = compiled
                self._compiled = True
            except Exception as exc:
                self._compile_error = str(exc)
                self.model = base_model
        else:
            self.model = base_model

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        for _ in range(4):
            self.inputs.append(torch.randn(self.batch_size, self.input_size, device=self.device, dtype=torch.float16))
            self.targets.append(torch.randn(self.batch_size, 1, device=self.device, dtype=torch.float16))
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.model is not None and self.optimizer is not None
        idx = self.batch_idx % len(self.inputs)
        self.batch_idx += 1
        with nvtx_range("optimized_dataparallel", enable=enable_nvtx):
            with torch.autocast("cuda", dtype=torch.float16):
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
        from common.python.benchmark_metrics import compute_memory_transfer_metrics
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
