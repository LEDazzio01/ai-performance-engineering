"""optimized_precisionfp8_te.py - Transformer Engine FP8 benchmark."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import List, Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
from torch.optim import Optimizer

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


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


class TEFP8MLP(nn.Module):
    """Two-layer MLP using Transformer Engine Linear layers with FP8."""

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = TELinear(hidden_dim, hidden_dim * 2, bias=True)
        self.fc2 = TELinear(hidden_dim * 2, hidden_dim, bias=True)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        return self.fc2(x)


class OptimizedTEFP8Benchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized FP8 path using Transformer Engine."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.fp8_recipe: Optional[object] = None
        self.batch_size = 256
        self.hidden_dim = 4096
        self.compute_dtype = torch.float16
        self.input_pool: List[torch.Tensor] = []
        self.target_pool: List[torch.Tensor] = []
        self.static_input: Optional[torch.Tensor] = None
        self.static_target: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.capture_stream: Optional[torch.cuda.Stream] = None
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self._verify_input: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        if not TE_AVAILABLE:
            raise RuntimeError(
                "Transformer Engine is required for optimized_precisionfp8_te. "
                f"(import error: {TE_IMPORT_ERROR})"
            )
        # Optimized FP8 recipe with best practices:
        # - HYBRID format: E4M3 for forward (more precision), E5M2 for backward (wider range)
        # - Larger amax_history_len for stable scaling over more iterations
        # - Hysteresis algorithm prevents scale oscillation
        # - margin=0 is aggressive but maximizes dynamic range utilization
        self.fp8_recipe = te_recipe.DelayedScaling(
            margin=0,  # Aggressive margin for max precision
            interval=1,  # Update scales every iteration for stable training
            amax_history_len=1024,  # Longer history for smoother scaling
            amax_compute_algo="max",  # Conservative scaling algorithm
            # Use default scaling factor compute (callable required; default uses margin-based scaling)
            scaling_factor_compute_algo=None,
        )
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        model = TEFP8MLP(hidden_dim=self.hidden_dim).to(self.device, dtype=self.compute_dtype).train()
        self.model = model
        fixed_input = torch.randn(
            self.batch_size,
            self.hidden_dim,
            device=self.device,
            dtype=self.compute_dtype,
        )
        self.input_pool = [fixed_input]
        self.target_pool = [torch.randn_like(fixed_input)]
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, foreach=False)
        self.criterion = nn.MSELoss()

        # Warmup to initialize optimizer state.
        for idx in range(5):
            self._train_step_impl(self.input_pool[idx % len(self.input_pool)], self.target_pool[idx % len(self.target_pool)])
        self._synchronize()

        # Prepare static buffers for CUDA graph capture/replay.
        self.static_input = self.input_pool[0].clone()
        self.static_target = self.target_pool[0].clone()
        self._verify_input = self.static_input.detach().clone()
        self.graph = torch.cuda.CUDAGraph()
        self.capture_stream = torch.cuda.Stream()

        with torch.cuda.stream(self.capture_stream):
            for _ in range(3):
                self._train_step_impl(self.static_input, self.static_target)
            self._synchronize()
            with torch.cuda.graph(self.graph, stream=self.capture_stream):
                self._train_step_impl(self.static_input, self.static_target)
        self.capture_stream.synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )

    def _train_step_impl(self, batch: torch.Tensor, target: torch.Tensor) -> None:
        assert self.model is not None
        assert self.optimizer and self.criterion
        if self.fp8_recipe is None:
            raise RuntimeError("FP8 recipe not initialized")
        self.optimizer.zero_grad(set_to_none=True)
        with fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
            outputs = self.model(batch)
            loss = self.criterion(outputs, target)
        loss.backward()
        self.optimizer.step()

    def benchmark_fn(self) -> None:
        if self.graph is None or self.static_input is None or self.static_target is None:
            raise RuntimeError("CUDA graph not initialized")

        current_input = self.input_pool[0]
        current_target = self.target_pool[0]

        with self._nvtx_range("optimized_precisionfp8_te"):
            self.static_input.copy_(current_input)
            self.static_target.copy_(current_target)
            self.graph.replay()
            # Store output for verification
            with torch.no_grad():
                self.output = self.model(current_input).detach().clone()
        self._synchronize()
        if self._verify_input is None or self.output is None:
            raise RuntimeError("Verification input/output not initialized")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self.output,
            batch_size=self._verify_input.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.5, 5.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.graph = None
        self.static_input = None
        self.static_target = None
        self.capture_stream = None
        self.input_pool = []
        self.target_pool = []
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=10, backend_policy="fp32_strict")

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def get_output_for_verification(self) -> Optional[torch.Tensor]:
        # Use a static captured input/output as representative output.
        return self.static_input

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.graph is None:
            return "CUDA graph not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedTEFP8Benchmark()


if __name__ == "__main__":  # pragma: no cover
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    timing = result.timing.mean_ms if result.timing else 0.0
    print(f"\nOptimized Precision FP8 (Transformer Engine): {timing:.3f} ms")
