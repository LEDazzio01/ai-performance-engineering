"""Shared helpers for Chapter 11 multi-stream benchmarks."""

from __future__ import annotations

from typing import List, Optional

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch11")
    return torch.device("cuda")


class StridedStreamBaseline(BaseBenchmark):
    """Baseline workload that executes strided copies on a single stream."""

    def __init__(
        self,
        nvtx_label: str,
        num_elements: int = 24_000_000,
        num_segments: int = 16,
    ):
        super().__init__()
        self.device = resolve_device()
        self.label = nvtx_label
        self.N = num_elements
        self.num_segments = num_segments
        self.stream = None
        self.host_input = None
        self.host_output = None
        self.host_in_chunks = None
        self.host_out_chunks = None
        self.device_chunks = None

    def setup(self) -> None:
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        self.host_input = torch.randn(
            self.N, device="cpu", dtype=torch.float32, pin_memory=True
        )
        self.host_output = torch.empty_like(self.host_input, pin_memory=True)
        self.host_in_chunks = list(torch.chunk(self.host_input, self.num_segments))
        self.host_out_chunks = list(torch.chunk(self.host_output, self.num_segments))
        self.device_chunks = [torch.empty_like(chunk, device=self.device) for chunk in self.host_in_chunks]
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range(self.label, enable=enable_nvtx):
            assert self.host_in_chunks is not None
            assert self.host_out_chunks is not None
            assert self.device_chunks is not None
            for h_in, h_out, d_buf in zip(self.host_in_chunks, self.host_out_chunks, self.device_chunks):
                with torch.cuda.stream(self.stream):
                    d_buf.copy_(h_in, non_blocking=True)
                    d_buf.mul_(2.0)
                    d_buf.add_(1.0)
                    d_buf.mul_(1.1)
                    d_buf.add_(0.5)
                    h_out.copy_(d_buf, non_blocking=True)
                # Naive path blocks on each segment, preventing overlap.
                self.stream.synchronize()

    def teardown(self) -> None:
        self.stream = None
        self.host_input = None
        self.host_output = None
        self.host_in_chunks = None
        self.host_out_chunks = None
        self.device_chunks = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        # Copy + compute per chunk is heavier; keep iteration count modest.
        return BenchmarkConfig(iterations=20, warmup=5)

    def validate_result(self) -> str | None:
        if self.host_output is None or self.host_input is None:
            return "Buffers not initialized"
        if self.host_output.shape != self.host_input.shape:
            return "Shape mismatch"
        if not torch.isfinite(self.host_output).all():
            return "Output contains non-finite values"
        return None

    def get_custom_metrics(self) -> Optional[dict]:
        """Return stream overlap metrics for the baseline (sequential) path."""
        bytes_transferred = float(self.N * 4 * 2)  # H2D + D2H
        return {
            f"{self.label}.elements": float(self.N),
            f"{self.label}.num_segments": float(self.num_segments),
            f"{self.label}.bytes_transferred": bytes_transferred,
            f"{self.label}.num_streams": 1.0,  # baseline uses 1 stream
            f"{self.label}.expected_overlap_pct": 0.0,
        }


class ConcurrentStreamOptimized(BaseBenchmark):
    """Optimized workload that splits data across multiple CUDA streams."""

    def __init__(
        self,
        nvtx_label: str,
        num_elements: int = 24_000_000,
        num_streams: int = 8,
        chunk_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = resolve_device()
        self.label = nvtx_label
        self.N = num_elements
        self.num_streams = num_streams
        self.dtype = chunk_dtype
        self.streams: List[torch.cuda.Stream] | None = None
        self.host_input: torch.Tensor | None = None
        self.host_output: torch.Tensor | None = None
        self.host_in_chunks: List[torch.Tensor] | None = None
        self.host_out_chunks: List[torch.Tensor] | None = None
        self.device_chunks: List[torch.Tensor] | None = None

    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        torch.manual_seed(42)
        self.host_input = torch.randn(
            self.N, dtype=self.dtype, device="cpu", pin_memory=True
        )
        self.host_output = torch.empty_like(self.host_input, pin_memory=True)
        chunks = torch.chunk(self.host_input, self.num_streams)
        if len(chunks) < self.num_streams:
            chunks = list(chunks)
            for _ in range(self.num_streams - len(chunks)):
                empty = torch.empty(0, dtype=self.dtype, device="cpu", pin_memory=True)
                chunks.append(empty)
        self.host_in_chunks = list(chunks)
        self.host_out_chunks = list(torch.chunk(self.host_output, len(self.host_in_chunks)))
        self.device_chunks = [torch.empty_like(chunk, device=self.device) for chunk in self.host_in_chunks]
        self.streams = [torch.cuda.Stream() for _ in self.device_chunks]
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range(self.label, enable=enable_nvtx):
            assert self.streams is not None
            assert self.host_in_chunks is not None
            assert self.host_out_chunks is not None
            assert self.device_chunks is not None
            with torch.no_grad():
                for stream, h_in, h_out, d_buf in zip(
                    self.streams, self.host_in_chunks, self.host_out_chunks, self.device_chunks
                ):
                    with torch.cuda.stream(stream):
                        if h_in.numel() == 0:
                            continue
                        d_buf.copy_(h_in, non_blocking=True)
                        d_buf.mul_(2.0)
                        d_buf.add_(1.0)
                        d_buf.mul_(1.1)
                        d_buf.add_(0.5)
                        h_out.copy_(d_buf, non_blocking=True)
                # Let async copies and compute overlap across streams.
                torch.cuda.synchronize()

    def teardown(self) -> None:
        self.streams = None
        self.host_input = None
        self.host_output = None
        self.host_in_chunks = None
        self.host_out_chunks = None
        self.device_chunks = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        # Streams overlap chunked work; fewer iterations keep runtime reasonable.
        return BenchmarkConfig(iterations=16, warmup=5)

    def validate_result(self) -> str | None:
        if not self.host_out_chunks:
            return "Chunks not initialized"
        for out in self.host_out_chunks:
            if not torch.isfinite(out).all():
                return "Output contains non-finite values"
        return None

    def get_custom_metrics(self) -> Optional[dict]:
        """Return stream overlap metrics for the optimized (concurrent) path."""
        bytes_transferred = float(self.N * 4 * 2)  # H2D + D2H
        return {
            f"{self.label}.elements": float(self.N),
            f"{self.label}.num_streams": float(self.num_streams),
            f"{self.label}.bytes_transferred": bytes_transferred,
            f"{self.label}.expected_overlap_pct": min(100.0, (self.num_streams - 1) / self.num_streams * 100),
        }
