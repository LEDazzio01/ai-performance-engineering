"""Shared PyTorch benchmark helpers for the Grace-Blackwell matmul demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig

TensorRunner = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class FeatureDescriptor:
    tag: str
    notes: str


class GraceBlackwellMatmulBenchmark(BaseBenchmark):
    """Benchmark wrapper that mimics the Modular.org Blackwell lessons."""

    def __init__(
        self,
        runner: TensorRunner,
        label: str,
        *,
        size: int = 2048,
        k_size: Optional[int] = None,
        iterations: int = 5,
        warmup: int = 5,
        timeout_seconds: int = 600,
        descriptor: Optional[FeatureDescriptor] = None,
        reference_runner: Optional[TensorRunner] = None,
    ) -> None:
        super().__init__()
        self._runner = runner
        self._label = label
        self._size_m = size
        self._size_n = size
        self._size_k = k_size if k_size is not None else size
        self._descriptor = descriptor
        self._reference_runner = reference_runner
        self._lhs: Optional[torch.Tensor] = None
        self._rhs: Optional[torch.Tensor] = None
        self._output: Optional[torch.Tensor] = None
        self._dtype = torch.float16
        self._reference: Optional[torch.Tensor] = None
        self._parameter_count = 0
        self._config = BenchmarkConfig(
            iterations=iterations,
            warmup=warmup,
            timeout_seconds=timeout_seconds,
            deterministic=False,
            enable_nvtx=True,
            enable_profiling=False,
        )
        self._config.measurement_timeout_seconds = max(
            timeout_seconds, self._config.measurement_timeout_seconds
        )
        self._config.warmup_timeout_seconds = max(
            timeout_seconds,
            self._config.warmup_timeout_seconds
            if self._config.warmup_timeout_seconds is not None
            else timeout_seconds,
        )
        self._config.profiling_timeout_seconds = max(
            2 * timeout_seconds,
            self._config.profiling_timeout_seconds
            if self._config.profiling_timeout_seconds is not None
            else 2 * timeout_seconds,
        )
        self._config.nsys_timeout_seconds = max(
            2 * timeout_seconds, self._config.nsys_timeout_seconds
        )
        self._config.ncu_timeout_seconds = max(
            2 * timeout_seconds, self._config.ncu_timeout_seconds
        )
        tokens_per_iteration = float(self._size_m * self._size_n)
        flops_per_iteration = float(2 * self._size_m * self._size_n * self._size_k)
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=tokens_per_iteration,
            samples_per_iteration=tokens_per_iteration,
            custom_units_per_iteration=flops_per_iteration,
            custom_unit_name="FLOPs",
        )
        self._workload_registered = True
        # Make it easy to fail fast on unsupported hardware from the runner layer.
        self.required_capabilities: dict[str, bool] = {}

    def setup(self) -> None:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: requires >=2 GPUs")
        device = self.device
        self._lhs = torch.randn(
            self._size_m, self._size_k, device=device, dtype=self._dtype
        )
        self._rhs = torch.randn(
            self._size_k, self._size_n, device=device, dtype=self._dtype
        )
        torch.cuda.synchronize(device)
        self._parameter_count = 0

        if self._reference_runner is not None:
            with torch.no_grad():
                ref = self._reference_runner(self._lhs, self._rhs)
            self._reference = ref.detach().clone()
            torch.cuda.synchronize(device)
        else:
            self._reference = None

    def benchmark_fn(self) -> None:
        assert self._lhs is not None and self._rhs is not None
        with self._nvtx_range(self._label):
            self._output = self._runner(self._lhs, self._rhs)
        torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        self._lhs = None
        self._rhs = None
        self._output = None
        super().teardown()

    def get_config(self) -> Optional[BenchmarkConfig]:
        return self._config

    def validate_result(self) -> Optional[str]:
        if self._reference is None or self._output is None:
            return None
        diff = (self._output - self._reference).abs()
        max_diff = diff.max().item()
        if torch.isnan(diff).any():
            return "NaNs detected in result tensor"
        if max_diff > 2.5:
            return f"Max abs diff {max_diff:.3f} exceeds tolerance 2.5"
        return None

    def get_problem_shape(self) -> tuple[int, int, int]:
        """Return the (M, N, K) dimensions for this GEMM."""
        return (self._size_m, self._size_n, self._size_k)

    @property
    def tensor_dtype(self) -> torch.dtype:
        """Data type used for the operands/results (defaults to FP16)."""
        return self._dtype

    def get_custom_metrics(self) -> Optional[dict[str, float]]:
        metrics: dict[str, float] = {}
        if self._descriptor is not None:
            metrics[f"feature_{self._descriptor.tag}_flag"] = 1.0
        return metrics or None

    def get_required_capabilities(self) -> Optional[dict[str, bool]]:
        """Override in subclasses to declare required device features."""
        return self.required_capabilities or None

    def get_verify_output(self) -> torch.Tensor:
        if self._output is None:
            raise RuntimeError("benchmark_fn() must be called before verification")
        return self._output.detach().clone()

    def get_verify_inputs(self) -> dict:
        if self._lhs is None or self._rhs is None:
            raise RuntimeError("setup() must be called before get_verify_inputs()")
        return {"lhs": self._lhs, "rhs": self._rhs}

    def get_input_signature(self) -> dict:
        if self._lhs is None or self._rhs is None:
            raise RuntimeError("setup() must be called before get_input_signature()")
        return {
            "label": self._label,
            "size_m": self._size_m,
            "size_n": self._size_n,
            "size_k": self._size_k,
            "shapes": {
                "lhs": tuple(self._lhs.shape),
                "rhs": tuple(self._rhs.shape),
            },
            "dtypes": {
                "lhs": str(self._lhs.dtype),
                "rhs": str(self._rhs.dtype),
            },
            "batch_size": int(self._size_m),
            "parameter_count": int(self._parameter_count),
            "precision_flags": {
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
        }

    def get_output_tolerance(self) -> tuple:
        return (1e-3, 5.0)

    @property
    def descriptor(self) -> Optional[FeatureDescriptor]:
        return self._descriptor
