"""baseline_stream_ordered.py - Serial execution on default stream."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


class BaselineStreamOrderedBenchmark(BaseBenchmark):
    """Sequential work on the default stream (no overlap)."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.host_requests: Optional[list[torch.Tensor]] = None
        self.host_outputs: Optional[list[torch.Tensor]] = None
        self.device_inputs: Optional[list[torch.Tensor]] = None
        self.device_outputs: Optional[list[torch.Tensor]] = None
        self.batch_size = 128
        self.hidden_dim = 1024
        self.num_streams = 4
        self.num_requests = 32  # More requests to amortize overhead

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        ).to(self.device).half().eval()

        self.host_requests = [
            torch.randn(
                self.batch_size, self.hidden_dim, device="cpu", dtype=torch.float16, pin_memory=True
            )
            for _ in range(self.num_requests)
        ]
        self.host_outputs = [
            torch.empty_like(req, device="cpu", pin_memory=True) for req in self.host_requests
        ]
        self.device_inputs = [
            torch.empty_like(req, device=self.device) for req in self.host_requests
        ]
        self.device_outputs = [
            torch.empty_like(inp, device=self.device) for inp in self.device_inputs
        ]
        self._synchronize()
        tokens = float(self.batch_size * self.hidden_dim * self.num_requests)
        self.register_workload_metadata(
            tokens_per_iteration=tokens,
            requests_per_iteration=float(self.num_requests),
        )

    def benchmark_fn(self) -> None:
        assert self.model is not None
        assert self.host_requests is not None
        assert self.host_outputs is not None
        assert self.device_inputs is not None
        assert self.device_outputs is not None

        with self._nvtx_range("stream_ordered"):
            with torch.no_grad():
                for h_req, h_out, d_in, d_out in zip(
                    self.host_requests, self.host_outputs, self.device_inputs, self.device_outputs
                ):
                    d_in.copy_(h_req, non_blocking=True)
                    out = self.model(d_in)
                    d_out.copy_(out)
                    h_out.copy_(d_out, non_blocking=True)
                    torch.cuda.synchronize()
        self._synchronize()

    def teardown(self) -> None:
        self.model = None
        self.host_requests = None
        self.host_outputs = None
        self.device_inputs = None
        self.device_outputs = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=80, warmup=8)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_stream_metrics
        return compute_stream_metrics(
            sequential_time_ms=getattr(self, '_sequential_ms', 10.0),
            overlapped_time_ms=getattr(self, '_overlapped_ms', 5.0),
            num_streams=getattr(self, 'num_streams', 4),
            num_operations=getattr(self, 'num_operations', 4),
        )

    def validate_result(self) -> Optional[str]:
        if self.host_outputs is None:
            return "Outputs not initialized"
        for out in self.host_outputs:
            if not torch.isfinite(out).all():
                return "Output contains non-finite values"
        return None


def get_benchmark() -> BaselineStreamOrderedBenchmark:
    return BaselineStreamOrderedBenchmark()
