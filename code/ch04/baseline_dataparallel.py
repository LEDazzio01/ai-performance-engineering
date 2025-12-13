"""baseline_dataparallel.py - DataParallel baseline (anti-pattern)."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.optim as optim


from typing import Optional

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from ch04.verification_payload_mixin import VerificationPayloadMixin


class SimpleNet(nn.Module):
    """Lightweight network to expose DataParallel + H2D overhead."""
    
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class BaselineDataParallelBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """DataParallel baseline - has GIL overhead and single-threaded execution."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.optimizer = None
        self.data = None
        self.target = None
        self._last_input: Optional[torch.Tensor] = None
        self._last_target: Optional[torch.Tensor] = None
        # Large input to make per-iteration H2D copies dominate.
        self.input_size = 4096
        self.batch_size = 4096
        self.output: Optional[torch.Tensor] = None
        # Training benchmarks don't support jitter check
        tokens = self.batch_size * self.input_size
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model, optimizer, and data."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: requires >=2 GPUs")
        torch.manual_seed(42)
        
        # Keep input tensors on CPU so DataParallel copies every iteration.
        model = SimpleNet(self.input_size).to(self.device)
        # IMPORTANT: Force DataParallel to a single device so baseline/optimized
        # remain comparable even on multi-GPU hosts.
        device_index = 0 if self.device.index is None else int(self.device.index)
        self.model = nn.DataParallel(model, device_ids=[device_index], output_device=device_index)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        data_gen = torch.Generator().manual_seed(1234)
        self.data = torch.randn(self.batch_size, self.input_size, generator=data_gen)
        self.target = torch.randn(self.batch_size, 1, generator=data_gen)

        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: DataParallel training step."""
        with self._nvtx_range("dataparallel"):
            gpu_data = self.data.to(self.device, non_blocking=False)
            gpu_target = self.target.to(self.device, non_blocking=False)
            self._last_input = gpu_data
            self._last_target = gpu_target
            output = self.model(gpu_data)
            loss = nn.functional.mse_loss(output, gpu_target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.output = output.detach()
        self._synchronize()

    def capture_verification_payload(self) -> None:
        if self._last_input is None or self._last_target is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        param_count = sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0
        self._set_verification_payload(
            inputs={"data": self._last_input, "target": self._last_target},
            output=self.output,
            batch_size=int(self.batch_size),
            parameter_count=param_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.optimizer = None
        self.data = None
        self.target = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
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
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.data is None:
            return "Data tensor not initialized"
        if self.target is None:
            return "Target tensor not initialized"
        # Check that model can produce valid output
        try:
            with torch.no_grad():
                test_output = self.model(self.data)
                if test_output.shape[0] != self.batch_size:
                    return f"Output batch size mismatch: expected {self.batch_size}, got {test_output.shape[0]}"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        return None



def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineDataParallelBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
