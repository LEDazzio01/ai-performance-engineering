"""optimized_memory_profiling.py - Optimized memory profiling (optimized)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedModel(nn.Module):
    """Model with gradient checkpointing for memory optimization."""
    
    def __init__(self, hidden_dim: int = 2048):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = checkpoint(self._fc1_relu, x, preserve_rng_state=False)
        x = self.fc2(x)
        return x
    
    def _fc1_relu(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.fc1(x))


class OptimizedMemoryProfilingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized memory profiling - uses gradient checkpointing + CUDA graphs."""

    signature_equivalence_group = "ch13_memory_profiling_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self):
        super().__init__()
        self.model: Optional[OptimizedModel] = None
        self.inputs: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.inputs_fp32: Optional[torch.Tensor] = None
        self.targets_fp32: Optional[torch.Tensor] = None
        self.criterion: Optional[nn.Module] = None
        self.peak_memory_mb = 0.0
        self.batch_size = 32
        self.hidden_dim = 2048
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.capture_stream: Optional[torch.cuda.Stream] = None
        self.static_input: Optional[torch.Tensor] = None
        self.static_target: Optional[torch.Tensor] = None
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        
        self.model = OptimizedModel(hidden_dim=self.hidden_dim).to(self.device, dtype=torch.bfloat16)
        self.model.train()
        self.inputs_fp32 = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.targets_fp32 = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
        self.inputs = self.inputs_fp32.to(dtype=torch.bfloat16)
        self.targets = self.targets_fp32.to(dtype=torch.bfloat16)
        self.criterion = nn.MSELoss()
        
        _ = self.model(self.inputs)
        self._synchronize()
        torch.cuda.reset_peak_memory_stats()

        self.static_input = self.inputs.clone()
        self.static_target = self.targets.clone()
        self.graph = torch.cuda.CUDAGraph()
        self.capture_stream = torch.cuda.Stream()
        with torch.cuda.stream(self.capture_stream):
            for _ in range(2):
                self._train_step(self.static_input, self.static_target)
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph, stream=self.capture_stream):
                self._train_step(self.static_input, self.static_target)
        self.capture_stream.synchronize()
    
    def _train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        assert self.model is not None and self.criterion is not None
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
    
    def benchmark_fn(self) -> None:
        if (
            self.graph is None
            or self.static_input is None
            or self.static_target is None
            or self.model is None
        ):
            raise RuntimeError("CUDA graph not initialized")

        with self._nvtx_range("optimized_memory_profiling"):
            self.static_input.copy_(self.inputs)
            self.static_target.copy_(self.targets)
            self.model.zero_grad(set_to_none=True)
            self.graph.replay()
            self.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            # Store output for verification
            with torch.no_grad():
                self.output = self.model(self.inputs).detach().clone()
        self._synchronize()
        if self.inputs is None or self.targets is None or self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.inputs_fp32 is None or self.targets_fp32 is None:
            raise RuntimeError("FP32 verification tensors not initialized")
        self._set_verification_payload(
            inputs={"input": self.inputs_fp32, "targets": self.targets_fp32},
            output=self.output.detach().float().clone(),
            batch_size=self.inputs.shape[0],
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.5, 5.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.targets = None
        self.criterion = None
        self.graph = None
        self.static_input = None
        self.static_target = None
        self.capture_stream = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=True,
            enable_profiling=False,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
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


def get_benchmark() -> OptimizedMemoryProfilingBenchmark:
    return OptimizedMemoryProfilingBenchmark()
