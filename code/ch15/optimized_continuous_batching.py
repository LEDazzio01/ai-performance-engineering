"""optimized_continuous_batching.py - Optimized continuous batching."""

from __future__ import annotations

from typing import Optional
import random

import torch
import torch.nn as nn

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch15.verification_payload_mixin import VerificationPayloadMixin


class OptimizedContinuousBatchingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: continuous batching with dynamic batch membership.

    Models a token-level scheduler that keeps the GPU batch full by:
    - Dropping finished requests immediately.
    - Pulling new requests into the active batch as slots open.
    """
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.samples: Optional[torch.Tensor] = None
        self.lengths: Optional[list[int]] = None
        self.lengths_tensor: Optional[torch.Tensor] = None
        self._verify_input: Optional[torch.Tensor] = None
        self.max_batch_size = 12
        self.hidden_dim = 1024
        self.num_samples = 144
        self.max_decode_steps = 32
        self.total_tokens = 0
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.num_samples),
            tokens_per_iteration=0.0,
        )
        self.output = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.num_samples),
            tokens_per_iteration=0.0,
        )
    
    def setup(self) -> None:
        """Setup: initialize model and request queue."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        ).to(self.device).eval()

        self.samples = torch.randn(self.num_samples, self.hidden_dim, device=self.device)
        self._verify_input = self.samples[:2].detach()

        rng = random.Random(123)
        self.lengths = [rng.randint(1, self.max_decode_steps) for _ in range(self.num_samples)]
        self.total_tokens = int(sum(self.lengths))
        self.lengths_tensor = torch.tensor(self.lengths, device=self.device, dtype=torch.int32)

        self.register_workload_metadata(
            requests_per_iteration=float(self.num_samples),
            tokens_per_iteration=float(self.total_tokens),
        )
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.num_samples),
            tokens_per_iteration=float(self.total_tokens),
        )
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: token-level scheduler keeps batches full."""
        assert self.model is not None and self.samples is not None and self.lengths is not None
        with self._nvtx_range("optimized_continuous_batching"):
            with torch.inference_mode():
                state = self.samples.clone()

                remaining = self.lengths.copy()
                active: list[int] = list(range(min(self.max_batch_size, self.num_samples)))
                next_idx = len(active)

                while active:
                    idx = torch.tensor(active, device=self.device, dtype=torch.int64)
                    batch_state = state.index_select(0, idx)
                    y = self.model(batch_state)
                    state.index_copy_(0, idx, y)

                    new_active: list[int] = []
                    for req_idx in active:
                        remaining[req_idx] -= 1
                        if remaining[req_idx] > 0:
                            new_active.append(req_idx)

                    active = new_active
                    while len(active) < self.max_batch_size and next_idx < self.num_samples:
                        active.append(next_idx)
                        next_idx += 1

                self.output = state

    def capture_verification_payload(self) -> None:
        if self.model is None or self._verify_input is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"probe": self._verify_input},
            output=self.output,
            batch_size=self.num_samples,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-2, 1e-2),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.samples = None
        self.lengths = None
        self.lengths_tensor = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=50.0,
            tpot_ms=10.0,
            total_tokens=float(self.total_tokens),
            total_requests=float(self.num_samples),
            batch_size=float(self.max_batch_size),
            max_batch_size=float(self.max_batch_size),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.samples is None:
            return "Samples not initialized"
        if self.lengths is None:
            return "Lengths not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedContinuousBatchingBenchmark()
