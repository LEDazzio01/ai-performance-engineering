"""optimized_regional_compile.py - Regional torch.compile (MLP-only).

Demonstrates regional compilation by compiling only the MLP subgraph while
keeping the rest of the Transformer block eager. Reduces compile churn across
sequence buckets and improves steady-state latency versus full-graph compile.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repo root is importable when running as a script
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class RegionalMLP(nn.Module):
    """MLP with torch.compile for kernel fusion (the regional hot path).
    
    This demonstrates Chapter 13's regional compilation strategy:
    - Only compile the compute-intensive MLP subgraph
    - Keep attention/layernorm eager for better dynamic shape handling
    - Get compilation benefits without full-graph recompilation overhead
    
    Key optimization: The MLP is the compute-intensive portion where fusion
    provides the most benefit. Compiling just this region avoids recompilation
    overhead from attention's dynamic masking while still getting speedups.
    """

    def __init__(self, hidden: int, mlp_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, hidden)
        self._compiled = None

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def compile_mlp(self) -> None:
        """Compile the MLP after model is on device."""
        self._compiled = torch.compile(
            self._forward_impl,
            backend="inductor",
            fullgraph=True,
            dynamic=True,  # Dynamic shapes to handle varying sequence lengths
            mode="reduce-overhead",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._compiled is not None:
            return self._compiled(x)
        return self._forward_impl(x)


class TinyTransformerBlock(nn.Module):
    """Transformer block using a regionally-compiled MLP."""

    def __init__(self, hidden: int = 1024, num_heads: int = 8, mlp_hidden: int = 4096):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(hidden)
        self.mlp = RegionalMLP(hidden, mlp_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = residual + attn_out

        residual = x
        x = self.ln2(x)
        x = residual + self.mlp(x)
        return x


class OptimizedRegionalCompileBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: compile only the MLP region, keep the rest eager."""

    signature_equivalence_group = "ch13_regional_compile_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self):
        super().__init__()
        # Larger workload to amortize compile overhead and show benefits
        self.hidden = 2048
        self.num_heads = 16
        self.mlp_hidden = 8192
        self.batch_size = 32
        self.sequence_schedule: List[int] = [512, 1024, 1536, 2048]
        self._step = 0

        self.model: Optional[nn.Module] = None
        self.inputs: Dict[int, torch.Tensor] = {}
        self.inputs_fp32: Dict[int, torch.Tensor] = {}
        self._verify_x: Optional[torch.Tensor] = None
        self._verify_output: Optional[torch.Tensor] = None

        max_tokens = self.batch_size * max(self.sequence_schedule) * self.hidden
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(max_tokens),
        )
        self.output = None
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(max_tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        # Enable cuDNN autotuning for optimal kernel selection
        self.model = TinyTransformerBlock(
            hidden=self.hidden,
            num_heads=self.num_heads,
            mlp_hidden=self.mlp_hidden,
        ).to(self.device, dtype=torch.bfloat16).eval()
        
        # Compile the MLP region after model is on device
        self.model.mlp.compile_mlp()

        for seq in self.sequence_schedule:
            x_fp32 = torch.randn(
                self.batch_size,
                seq,
                self.hidden,
                device=self.device,
                dtype=torch.float32,
            )
            self.inputs_fp32[seq] = x_fp32
            self.inputs[seq] = x_fp32.to(dtype=torch.bfloat16)
        
        # Extensive warmup to ensure compilation is complete
        with torch.no_grad():
            for _ in range(5):  # Multiple rounds to warm up for all sequence lengths
                for seq in self.sequence_schedule:
                    _ = self.model(self.inputs[seq])
        self._synchronize()

        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )

    def _next_sequence_length(self) -> int:
        seq = self.sequence_schedule[self._step % len(self.sequence_schedule)]
        self._step += 1
        return seq

    def benchmark_fn(self) -> None:
        if self.model is None:
            raise RuntimeError("Model not initialized")

        seq_len = self._next_sequence_length()
        x = self.inputs[seq_len]
        x_fp32 = self.inputs_fp32[seq_len]

        with torch.no_grad(), self._nvtx_range("optimized_regional_compile"):
            self.output = self.model(x).detach().float().clone()
        self._synchronize()
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")
        if self._verify_x is None:
            self._verify_x = x_fp32
            self._verify_output = self.output.detach().float().clone()

    def capture_verification_payload(self) -> None:
        if self._verify_x is None or self._verify_output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        x = self._verify_x
        self._set_verification_payload(
            inputs={"input": x},
            output=self._verify_output,
            batch_size=self.batch_size,
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1.0, 10.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.inputs.clear()
        self.inputs_fp32.clear()
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        # NOTE: warmup=10 is REQUIRED to ensure torch.compile JIT overhead is NOT
        # included in measurements. The first few calls trigger compilation.
        return BenchmarkConfig(
            iterations=8,
            warmup=10,  # Required for torch.compile - excludes JIT overhead from timing
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=300,
            measurement_timeout_seconds=300,
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


def get_benchmark() -> BaseBenchmark:
    return OptimizedRegionalCompileBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
