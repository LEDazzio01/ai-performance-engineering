"""Shared helpers for Chapter 11 multi-stream benchmarks."""

from __future__ import annotations

from typing import List

import torch

from common.python.benchmark_harness import Benchmark, BenchmarkConfig
from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch11")
    return torch.device("cuda")


class StridedStreamBaseline(Benchmark):
    """Baseline workload that executes strided copies on a single stream."""

    def __init__(
        self,
        nvtx_label: str,
        num_elements: int = 8_000_000,
        num_segments: int = 8,
    ):
        self.device = resolve_device()
        self.label = nvtx_label
        self.N = num_elements
        self.num_segments = num_segments
        self.stream = None
        self.input = None
        self.output = None

    def setup(self) -> None:
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty_like(self.input)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range(self.label, enable=enable_nvtx):
            with torch.cuda.stream(self.stream):
                for offset in range(self.num_segments):
                    segment = slice(offset, None, self.num_segments)
                    self.output[segment] = self.input[segment] * 2.0 + 1.0
            torch.cuda.synchronize()

    def teardown(self) -> None:
        self.stream = None
        self.input = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=60, warmup=10)

    def validate_result(self) -> str | None:
        if self.output is None or self.input is None:
            return "Buffers not initialized"
        if self.output.shape != self.input.shape:
            return "Shape mismatch"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None


class ConcurrentStreamOptimized(Benchmark):
    """Optimized workload that splits data across multiple CUDA streams."""

    def __init__(
        self,
        nvtx_label: str,
        num_elements: int = 8_000_000,
        num_streams: int = 8,
        chunk_dtype: torch.dtype = torch.float16,
    ):
        self.device = resolve_device()
        self.label = nvtx_label
        self.N = num_elements
        self.num_streams = num_streams
        self.dtype = chunk_dtype
        self.streams: List[torch.cuda.Stream] | None = None
        self.data_chunks: List[torch.Tensor] | None = None
        self.output_chunks: List[torch.Tensor] | None = None

    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        torch.manual_seed(42)
        base = torch.randn(self.N, dtype=self.dtype, device=self.device)
        chunks = torch.chunk(base, self.num_streams)
        # torch.chunk might create fewer-than-requested chunks if N < num_streams.
        if len(chunks) < self.num_streams:
            chunks = list(chunks)
            for _ in range(self.num_streams - len(chunks)):
                chunks.append(torch.empty(0, dtype=self.dtype, device=self.device))
        self.data_chunks = list(chunks)
        self.output_chunks = [torch.empty_like(chunk) for chunk in self.data_chunks]
        self.streams = [torch.cuda.Stream() for _ in self.data_chunks]
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range(self.label, enable=enable_nvtx):
            assert self.streams is not None
            assert self.data_chunks is not None
            assert self.output_chunks is not None
            with torch.no_grad():
                for stream, inp, out in zip(self.streams, self.data_chunks, self.output_chunks):
                    with torch.cuda.stream(stream):
                        if inp.numel() == 0:
                            continue
                        out.copy_(inp * 2.0 + 1.0)
                for stream in self.streams:
                    stream.synchronize()

    def teardown(self) -> None:
        self.streams = None
        self.data_chunks = None
        self.output_chunks = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=40, warmup=8)

    def validate_result(self) -> str | None:
        if not self.output_chunks:
            return "Chunks not initialized"
        for out in self.output_chunks:
            if not torch.isfinite(out).all():
                return "Output contains non-finite values"
        return None
