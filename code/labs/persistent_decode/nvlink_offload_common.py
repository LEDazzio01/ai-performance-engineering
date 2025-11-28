"""Shared NVLink offload helper for the persistent decode lab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


@dataclass
class OffloadConfig:
    use_pinned: bool
    non_blocking: bool
    use_copy_stream: bool
    batch_size: int = 2
    num_layers: int = 2
    num_heads: int = 8
    head_dim: int = 64
    max_seq_len: int = 2048
    chunk_tokens: int = 256
    dtype: torch.dtype = torch.float16


class NvlinkOffloadBenchmark(BaseBenchmark):
    """Measure H2D/D2H swap time for a KV-cache slice."""

    def __init__(self, cfg: Optional[OffloadConfig] = None, label: str = "nvlink_offload"):
        super().__init__()
        self.cfg = cfg or OffloadConfig(use_pinned=False, non_blocking=False, use_copy_stream=False)
        self.label = label

        self.cpu_cache: Optional[torch.Tensor] = None
        self.gpu_cache: Optional[torch.Tensor] = None
        self.copy_stream: Optional[torch.cuda.Stream] = None
        self.next_start: int = 0
        self._bytes_per_iteration: float = 0.0

    def setup(self) -> None:
        torch.manual_seed(2025)
        shape = (
            self.cfg.num_layers,
            2,  # k/v
            self.cfg.batch_size,
            self.cfg.num_heads,
            self.cfg.max_seq_len,
            self.cfg.head_dim,
        )

        self.cpu_cache = torch.zeros(shape, dtype=self.cfg.dtype, pin_memory=self.cfg.use_pinned)
        self.gpu_cache = torch.zeros(shape, dtype=self.cfg.dtype, device=self.device)
        self.copy_stream = torch.cuda.Stream() if self.cfg.use_copy_stream else None

        elements_per_chunk = (
            self.cfg.num_layers
            * 2
            * self.cfg.batch_size
            * self.cfg.num_heads
            * self.cfg.chunk_tokens
            * self.cfg.head_dim
        )
        # H2D + D2H each iteration
        self._bytes_per_iteration = elements_per_chunk * self.cfg.dtype.itemsize * 2.0
        self.register_workload_metadata(bytes_per_iteration=self._bytes_per_iteration)

    def benchmark_fn(self) -> None:
        assert self.cpu_cache is not None and self.gpu_cache is not None

        start = self.next_start
        end = min(start + self.cfg.chunk_tokens, self.cfg.max_seq_len)
        slice_len = end - start
        if slice_len <= 0:
            self.next_start = 0
            return

        cpu_slice = self.cpu_cache[..., start:end, :]
        if self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                self.gpu_cache[..., :slice_len, :].copy_(cpu_slice.to(self.device, non_blocking=self.cfg.non_blocking))
            torch.cuda.current_stream().wait_stream(self.copy_stream)
        else:
            self.gpu_cache[..., :slice_len, :].copy_(cpu_slice.to(self.device, non_blocking=self.cfg.non_blocking))

        # Lightweight compute to keep the slice "hot"
        self.gpu_cache[..., :slice_len, :].mul_(1.0001)

        if self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                target = self.cpu_cache[..., start:end, :]
                target.copy_(self.gpu_cache[..., :slice_len, :].to("cpu", non_blocking=self.cfg.non_blocking))
            torch.cuda.current_stream().wait_stream(self.copy_stream)
        else:
            target = self.cpu_cache[..., start:end, :]
            target.copy_(self.gpu_cache[..., :slice_len, :].to("cpu", non_blocking=self.cfg.non_blocking))

        self.next_start = 0 if end >= self.cfg.max_seq_len else end

    def teardown(self) -> None:
        self.cpu_cache = None
        self.gpu_cache = None
        self.copy_stream = None
        super().teardown()

    def get_config(self) -> Optional[BenchmarkConfig]:
        # Keep iterations small to avoid large host memory pressure during sweeps.
        return BenchmarkConfig(
            iterations=16,
            warmup=5,
            timeout_seconds=180,
            measurement_timeout_seconds=180,
            use_subprocess=False,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return NVLink offload performance metrics."""
        return {
            f"{self.label}.bytes_per_iteration": self._bytes_per_iteration,
            f"{self.label}.batch_size": float(self.cfg.batch_size),
            f"{self.label}.num_layers": float(self.cfg.num_layers),
            f"{self.label}.chunk_tokens": float(self.cfg.chunk_tokens),
            f"{self.label}.use_pinned": float(self.cfg.use_pinned),
            f"{self.label}.non_blocking": float(self.cfg.non_blocking),
        }
