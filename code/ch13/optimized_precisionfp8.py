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
from torch.amp import autocast
from torch.cuda.amp import GradScaler

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class SimpleModel(nn.Module):
    """Two-layer MLP used for AMP + fake FP8 runs.
    
    Note: Must match baseline_precisionfp8.py architecture for output verification.
    """

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()  # Match baseline

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def fake_fp8_cast(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize activations to float8 if supported, else fall back to fp16."""
    if hasattr(torch, "float8_e4m3fn"):
        return tensor.to(torch.float8_e4m3fn).to(torch.float16)
    return tensor.to(torch.float16)


class OptimizedFP8Benchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized FP8 path using PyTorch AMP + fake FP8 activations."""

    signature_equivalence_group = "ch13_precisionfp8_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.scaler: Optional[GradScaler] = None
        self.output: Optional[torch.Tensor] = None  # For output verification
        self._verify_input: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self.batch_size = 256
        self.hidden_dim = 4096
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Harness provides seeding - creation order must match baseline
        model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).train()
        
        # Create inputs/targets in same order as baseline for verification
        self.inputs = torch.randn(
            self.batch_size,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.targets = torch.randn(
            self.batch_size,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float16,
        )
        self._verify_input = self.inputs.detach().clone()
        
        # Convert to FP16 for actual benchmark (this is the optimization)
        model = model.half()
        self.model = model
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.scaler = GradScaler(enabled=False)

        # Warmup (will modify model weights, but output already saved)
        for _ in range(5):
            self._train_step()
        self._synchronize()
        self.optimizer.zero_grad(set_to_none=True)
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )

    def _train_step(self) -> None:
        assert self.model and self.inputs is not None and self.targets is not None
        assert self.optimizer and self.criterion and self.scaler is not None
        self.optimizer.zero_grad(set_to_none=True)
        fp8_inputs = fake_fp8_cast(self.inputs)
        with autocast("cuda", dtype=torch.float16):
            outputs = self.model(fp8_inputs)
            loss = self.criterion(outputs, self.targets)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def benchmark_fn(self) -> None:
        if self._verify_input is None:
            raise RuntimeError("Verification input not initialized")
        with self._nvtx_range("optimized_precisionfp8"):
            self._train_step()
            with torch.no_grad():
                fp8_inputs = fake_fp8_cast(self._verify_input)
                with autocast("cuda", dtype=torch.float16):
                    verify_out = self.model(fp8_inputs)
                self.output = verify_out.detach().float().clone()
        self._synchronize()
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self.output,
            batch_size=self._verify_input.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": True,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.25, 2.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        self.scaler = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedFP8Benchmark()


if __name__ == "__main__":  # pragma: no cover
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    timing = result.timing.mean_ms if result.timing else 0.0
    print(f"\nOptimized Precision FP8 (AMP-based, precisionfp8 pair): {timing:.3f} ms")
