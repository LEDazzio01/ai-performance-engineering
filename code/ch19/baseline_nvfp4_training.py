"""BF16 reference training run to compare against NVFP4/Transformer Engine paths."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class _BF16Trainer(nn.Module):
    """Simple transformer-style feed-forward stack to stress activations."""

    def __init__(self, hidden_dim: int, intermediate_dim: int, num_layers: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.LayerNorm(hidden_dim, eps=1e-5),
                    nn.Linear(hidden_dim, intermediate_dim, bias=True),
                    nn.GELU(),
                    nn.Linear(intermediate_dim, hidden_dim, bias=True),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - standard forward
        return self.net(x)


class BaselineNVFP4TrainingBenchmark(BaseBenchmark):
    """Reference BF16 training loop (no NVFP4 compression)."""

    def __init__(self) -> None:
        super().__init__()
        # Larger workload to amortize TE overhead and show NVFP4 benefits
        self.hidden_dim = 4096
        self.intermediate_dim = self.hidden_dim * 4
        self.num_layers = 8
        self.batch_size = 32
        self.seq_len = 1024
        self.micro_batches = 4
        self.model: Optional[_BF16Trainer] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.inputs: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        tokens = self.batch_size * self.seq_len * self.micro_batches
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.micro_batches),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        self.model = _BF16Trainer(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
        ).to(self.device, dtype=torch.bfloat16)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, fused=True)
        self.inputs = [
            torch.randn(
                self.batch_size,
                self.seq_len,
                self.hidden_dim,
                device=self.device,
                dtype=torch.bfloat16,
            )
            for _ in range(self.micro_batches)
        ]
        self.targets = [torch.randn_like(self.inputs[0]) for _ in range(self.micro_batches)]
        torch.cuda.synchronize(self.device)

    def _train_step(self, idx: int) -> None:
        assert self.model is not None and self.optimizer is not None
        inp = self.inputs[idx]
        target = self.targets[idx]

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model(inp)
            loss = F.mse_loss(out, target)
        loss.backward()
        self.optimizer.step()

    def benchmark_fn(self) -> None:
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("nvfp4_training_baseline", enable=enable_nvtx):
            for idx in range(self.micro_batches):
                self._train_step(idx)
        torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        self.model = None
        self.optimizer = None
        self.inputs = []
        self.targets = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=5,
            enable_memory_tracking=False,
            deterministic=False,  # allow fastest kernels
            seed=None,  # avoid global seeding that can trip TE/cuRAND
            measurement_timeout_seconds=60,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if not self.inputs:
            return None
        return {
            "nvfp4_baseline.micro_batches": float(self.micro_batches),
            "nvfp4_baseline.seq_len": float(self.seq_len),
        }

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.optimizer is None:
            return "Trainer not initialized"
        if not self.inputs:
            return "Input tensors missing"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineNVFP4TrainingBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(f"BF16 baseline mean: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
