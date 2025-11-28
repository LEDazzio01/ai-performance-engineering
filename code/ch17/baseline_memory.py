"""baseline_memory.py - Baseline standard GPU memory allocation."""

from __future__ import annotations

from typing import Optional
import warnings

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

BATCH_SIZE = 512
INPUT_DIM = 2048
HIDDEN_DIM = 2048
REPETITIONS = 8


class BaselineMemoryBenchmark(BaseBenchmark):
    """Baseline: Standard GPU memory allocation."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.batch_size = BATCH_SIZE
        self.input_dim = INPUT_DIM
        self.repetitions = REPETITIONS
        self.host_batches: list[torch.Tensor] = []
        self._prev_threads: Optional[int] = None
        self._prev_interop_threads: Optional[int] = None
        self._threads_overridden = False
        self._interop_overridden = False
        tokens = self.batch_size * self.input_dim * self.repetitions
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repetitions),
            tokens_per_iteration=float(tokens),
        )
    
    @staticmethod
    def _safe_set_thread_fn(setter, value: int, label: str, warn=True) -> bool:
        """Try to set a torch threading knob without aborting the benchmark."""
        try:
            setter(value)
            return True
        except RuntimeError as err:
            if warn:
                warnings.warn(f"Unable to set {label} (continuing with defaults): {err}")
            return False
    
    def setup(self) -> None:
        torch.manual_seed(42)
        self._prev_threads = torch.get_num_threads()
        self._prev_interop_threads = torch.get_num_interop_threads()
        self._threads_overridden = self._safe_set_thread_fn(
            torch.set_num_threads, 1, "num_threads"
        )
        self._interop_overridden = self._safe_set_thread_fn(
            torch.set_num_interop_threads, 1, "num_interop_threads"
        )
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, self.input_dim),
        ).to(self.device, dtype=torch.float32).eval()
        self.host_batches = [
            torch.randint(
                0,
                256,
                (self.batch_size, self.input_dim),
                device="cpu",
                dtype=torch.uint8,
            )
            for _ in range(self.repetitions)
        ]
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        if self.model is None:
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("baseline_memory"):
            with torch.no_grad():
                for compressed in self.host_batches:
                    host_batch = compressed.to(dtype=torch.float32)
                    host_batch.mul_(1.0 / 255.0)
                    host_batch.add_(-0.5)
                    host_batch.mul_(2.0)
                    host_batch.tanh_()
                    device_batch = host_batch.to(self.device, dtype=torch.float32, non_blocking=False)
                    _ = self.model(device_batch)
        self._synchronize()
    
    def teardown(self) -> None:
        if self._threads_overridden and self._prev_threads is not None:
            self._safe_set_thread_fn(torch.set_num_threads, self._prev_threads, "num_threads", warn=False)
        if self._interop_overridden and self._prev_interop_threads is not None:
            self._safe_set_thread_fn(torch.set_num_interop_threads, self._prev_interop_threads, "num_interop_threads", warn=False)
        self.model = None
        self.host_batches = []
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=getattr(self, '_ttft_ms', 50.0),
            tpot_ms=getattr(self, '_tpot_ms', 10.0),
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaselineMemoryBenchmark:
    return BaselineMemoryBenchmark()
