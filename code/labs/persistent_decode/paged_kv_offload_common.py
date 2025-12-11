"""Paged/NVMe-style KV-cache offload microbenchmarks with fused FP8 gating.

This module provides a small, GPU-backed benchmark that simulates:
1) A hot KV window that remains resident on the GPU.
2) Cold KV pages that live in either pageable CPU memory or an NVMe-backed
   memmap file.
3) Optional pinned staging + async H2D copies for overlap.
4) FP8 KV usage that is only enabled when a fused FlashAttention-style path is
   likely to exist (B200/GB200 or newer).

The goal is to encode the practical rule from the post:
- Use FP8 KV only when a fused attention kernel is available; otherwise fall
  back to FP16 to avoid paying dequant cost with no speedup.
- Use paged/NVMe-style offload when context length forces it, and measure the
  TTFT impact of pulling pages back in.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig

try:  # Prefer the Blackwell-aware SDPA selector when available.
    from arch_config import prefer_sdpa_backends  # type: ignore
except Exception:  # pragma: no cover - defensive import
    prefer_sdpa_backends = None  # type: ignore




def _supports_fp8_kv() -> bool:
    """Return True if FP8 KV is even representable in this build of PyTorch."""
    return hasattr(torch, "float8_e4m3fn") and torch.cuda.is_available()


def _supports_fused_fp8_attention() -> bool:
    """Heuristic: Blackwell/Grace-Blackwell GPUs (SM 10.x or 12.x) + flash/TE SDP."""
    if not torch.cuda.is_available():
        return False
    cc_major, _ = torch.cuda.get_device_capability()
    if cc_major >= 10:
        return True
    # For completeness, honor Transformer Engine SDPA when available on older GPUs.
    try:
        from torch.nn.attention import sdpa_kernel as _sdpa_kernel, SDPBackend as _SDPBackend

        available = getattr(_sdpa_kernel, "available_backends", lambda: [])()
        te_backend = getattr(_SDPBackend, "TRANSFORMER_ENGINE", None)
        if available and te_backend is not None:
            return te_backend in available
    except Exception:
        pass
    return False


def _randn_safe(*shape: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Generate random numbers; fail fast if the target dtype lacks a normal kernel.
    """
    try:
        return torch.randn(*shape, device=device, dtype=dtype)
    except (RuntimeError, NotImplementedError) as exc:
        msg = str(exc).lower()
        float8_e4m3 = getattr(torch, "float8_e4m3fn", None)
        float8_e5m2 = getattr(torch, "float8_e5m2fn", None)
        is_float8 = dtype in (float8_e4m3, float8_e5m2)
        if is_float8 and ("not implemented" in msg or "normal_kernel" in msg):
            raise RuntimeError(
                f"FP8 normal kernel unavailable for dtype {dtype}; aborting benchmark."
            ) from exc
        raise


def _np_dtype_for(torch_dtype: torch.dtype) -> np.dtype:
    """Map a torch dtype to a numpy dtype used for the memmap backing store."""
    float8_e4m3 = getattr(torch, "float8_e4m3fn", None)
    float8_e5m2 = getattr(torch, "float8_e5m2fn", None)
    if torch_dtype in {float8_e4m3, float8_e5m2}:
        # memmap sticks to fp16; conversion to fp8 happens during staging/H2D.
        return np.float16
    return torch.empty([], dtype=torch_dtype).numpy().dtype


@dataclass
class PagedKVConfig:
    """Configuration for paged KV-cache simulation."""

    batch_size: int = 2
    num_heads: int = 16
    head_dim: int = 128
    max_seq_len: int = 8192
    page_tokens: int = 512
    decode_tokens: int = 64
    use_pinned_stage: bool = False
    use_async_stream: bool = False
    use_memmap: bool = False  # When True, store cold pages on disk to mimic NVMe.
    prefer_fp8: bool = True
    require_fused_fp8: bool = False  # If True, FP8 is only used when fused path is present.
    fallback_dtype: torch.dtype = torch.float16
    prefetch_next_page: bool = False


class PagedKVOffloadBenchmark(BaseBenchmark):
    """Synthetic decode microbenchmark with paged KV offload and FP8 gating."""

    def __init__(self, cfg: Optional[PagedKVConfig] = None, label: str = "paged_kv_offload"):
        super().__init__()
        self.cfg = cfg or PagedKVConfig()
        self.label = label

        self.runtime_dtype: torch.dtype = self.cfg.fallback_dtype
        self.enable_flash: bool = False
        self._fp8_reason: str = ""

        self.hot_k: Optional[torch.Tensor] = None
        self.hot_v: Optional[torch.Tensor] = None
        self.staging: Optional[torch.Tensor] = None
        self.prefetch_staging: Optional[torch.Tensor] = None
        self.prefetched_range: Optional[Tuple[int, int]] = None
        self.copy_stream: Optional[torch.cuda.Stream] = None

        self.host_cache: Optional[torch.Tensor] = None
        self.host_memmap: Optional[np.memmap] = None
        self._memmap_path: Optional[Path] = None

        self.page_cursor: int = 0
        self._bytes_per_iteration: float = 0.0
        self.output: Optional[torch.Tensor] = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    # -------------------- Setup helpers --------------------

    def _select_runtime_dtype(self) -> torch.dtype:
        if self.cfg.prefer_fp8 and _supports_fp8_kv():
            if not _supports_fused_fp8_attention():
                raise RuntimeError("SKIPPED: FP8 requested but fused attention backend is unavailable.")
            self._fp8_reason = "Using FP8 KV: fused FlashAttention path detected."
            return torch.float8_e4m3fn  # type: ignore[attr-defined]
        return self.cfg.fallback_dtype

    def _init_host_cache(self, shape: Tuple[int, ...]) -> None:
        if self.cfg.use_memmap:
            np_dtype = _np_dtype_for(self.runtime_dtype)
            tmp_dir = Path(tempfile.mkdtemp(prefix="paged_kv_cache_"))
            self._memmap_path = tmp_dir / "kv_cache.bin"
            self.host_memmap = np.memmap(self._memmap_path, mode="w+", dtype=np_dtype, shape=shape)
            self.host_memmap[:] = np.random.randn(*shape).astype(np_dtype)
        else:
            self.host_cache = torch.randn(
                shape,
                dtype=torch.float16,
                pin_memory=self.cfg.use_pinned_stage,
            )

    def _stage_page(self, start: int, into_prefetch: bool = False) -> Tuple[torch.Tensor, int]:
        end = min(start + self.cfg.page_tokens, self.cfg.max_seq_len)
        slice_len = end - start
        target = self.prefetch_staging if into_prefetch else self.staging
        assert target is not None

        if self.host_memmap is not None:
            np_slice = self.host_memmap[..., start:end, :]
            target[..., :slice_len, :].copy_(torch.from_numpy(np_slice))
        elif self.host_cache is not None:
            target[..., :slice_len, :].copy_(self.host_cache[..., start:end, :])
        else:
            raise RuntimeError("Host cache not initialized")
        return target, slice_len

    def _copy_to_device(self, staged: torch.Tensor, slice_len: int) -> None:
        assert self.hot_k is not None and self.hot_v is not None
        if self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                self.hot_k[..., :slice_len, :].copy_(
                    staged[0, ..., :slice_len, :].to(self.device, dtype=self.runtime_dtype, non_blocking=True)
                )
                self.hot_v[..., :slice_len, :].copy_(
                    staged[1, ..., :slice_len, :].to(self.device, dtype=self.runtime_dtype, non_blocking=True)
                )
            torch.cuda.current_stream().wait_stream(self.copy_stream)
        else:
            self.hot_k[..., :slice_len, :].copy_(
                staged[0, ..., :slice_len, :].to(self.device, dtype=self.runtime_dtype, non_blocking=False)
            )
            self.hot_v[..., :slice_len, :].copy_(
                staged[1, ..., :slice_len, :].to(self.device, dtype=self.runtime_dtype, non_blocking=False)
            )

    def setup(self) -> None:
        torch.manual_seed(7)
        self.runtime_dtype = self._select_runtime_dtype()
        self.enable_flash = _supports_fused_fp8_attention() or self.runtime_dtype in (torch.float16, torch.bfloat16)

        head_shape = (
            self.cfg.batch_size,
            self.cfg.num_heads,
            self.cfg.page_tokens,
            self.cfg.head_dim,
        )
        self.hot_k = torch.zeros(head_shape, device=self.device, dtype=self.runtime_dtype)
        self.hot_v = torch.zeros_like(self.hot_k)

        staging_dtype = torch.float16
        staging_shape = (
            2,  # k and v planes
            self.cfg.batch_size,
            self.cfg.num_heads,
            self.cfg.page_tokens,
            self.cfg.head_dim,
        )
        self.staging = torch.empty(
            staging_shape,
            device="cpu",
            dtype=staging_dtype,
            pin_memory=self.cfg.use_pinned_stage,
        )
        if self.cfg.prefetch_next_page:
            self.prefetch_staging = torch.empty(
                staging_shape,
                device="cpu",
                dtype=staging_dtype,
                pin_memory=self.cfg.use_pinned_stage,
            )
        self.copy_stream = torch.cuda.Stream() if self.cfg.use_async_stream else None

        host_shape = (
            2,  # k and v
            self.cfg.batch_size,
            self.cfg.num_heads,
            self.cfg.max_seq_len,
            self.cfg.head_dim,
        )
        self._init_host_cache(host_shape)

        bytes_per_page = (
            2
            * self.cfg.batch_size
            * self.cfg.num_heads
            * self.cfg.page_tokens
            * self.cfg.head_dim
            * torch.finfo(self.runtime_dtype).bits
            / 8.0
        )
        self._bytes_per_iteration = float(bytes_per_page)
        self.register_workload_metadata(bytes_per_iteration=self._bytes_per_iteration)

    # -------------------- Benchmark --------------------

    def _maybe_use_prefetch(self, start: int) -> Optional[Tuple[torch.Tensor, int]]:
        if self.prefetched_range is None or self.prefetch_staging is None:
            return None
        pref_start, pref_end = self.prefetched_range
        if start != pref_start:
            return None
        return self.prefetch_staging, pref_end - pref_start

    def benchmark_fn(self) -> None:
        start = self.page_cursor
        prefetched = self._maybe_use_prefetch(start)
        if prefetched is not None:
            staged, slice_len = prefetched
        else:
            staged, slice_len = self._stage_page(start)
        self._copy_to_device(staged, slice_len)

        # Simple attention step that will pick flash/mathematics based on dtype/backend.
        q = _randn_safe(
            self.cfg.batch_size,
            self.cfg.num_heads,
            self.cfg.decode_tokens,
            self.cfg.head_dim,
            device=self.device,
            dtype=self.runtime_dtype,
        )
        k = self.hot_k[..., :slice_len, :]
        v = self.hot_v[..., :slice_len, :]
        ctx = prefer_sdpa_backends() if prefer_sdpa_backends is not None else nullcontext()
        with ctx:
            attn_out = F.scaled_dot_product_attention(q, k, v)

        # Queue prefetch for the next page if requested.
        if self.cfg.prefetch_next_page and self.prefetch_staging is not None:
            next_start = (start + self.cfg.page_tokens) % self.cfg.max_seq_len
            staged_prefetch, pref_len = self._stage_page(next_start, into_prefetch=True)
            self.prefetched_range = (next_start, next_start + pref_len)
        else:
            self.prefetched_range = None

        self.page_cursor = (start + self.cfg.page_tokens) % self.cfg.max_seq_len
        # Capture a slice of attention output for verification
        self.output = attn_out[:, :, :1, : min(8, attn_out.shape[-1])].detach().float().clone()

    # -------------------- Teardown --------------------

    def teardown(self) -> None:
        self.hot_k = None
        self.hot_v = None
        self.staging = None
        self.prefetch_staging = None
        self.copy_stream = None
        self.host_cache = None
        if self.host_memmap is not None:
            self.host_memmap._mmap.close()  # type: ignore[attr-defined]
        self.host_memmap = None
        if self._memmap_path is not None:
            try:
                os.remove(self._memmap_path)
                os.rmdir(self._memmap_path.parent)
            except OSError:
                pass
        self._memmap_path = None
        self.output = None
        super().teardown()

    # -------------------- Harness config --------------------

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=12,
            warmup=5,
            timeout_seconds=300,
            measurement_timeout_seconds=300,
            deterministic=False,
            use_subprocess=False,
        )

    def validate_result(self) -> Optional[str]:
        if self._fp8_reason and self.cfg.prefer_fp8:
            # Report the path we took for visibility; not a failure.
            print(f"[{self.label}] {self._fp8_reason}")
        return None

    def get_custom_metrics(self) -> Optional[dict]:
        """Return paged KV offload performance metrics."""
        return {
            f"{self.label}.bytes_per_iteration": self._bytes_per_iteration,
            f"{self.label}.batch_size": float(self.cfg.batch_size),
            f"{self.label}.num_heads": float(self.cfg.num_heads),
            f"{self.label}.head_dim": float(self.cfg.head_dim),
            f"{self.label}.page_tokens": float(self.cfg.page_tokens),
            f"{self.label}.decode_tokens": float(self.cfg.decode_tokens),
            f"{self.label}.max_seq_len": float(self.cfg.max_seq_len),
            f"{self.label}.use_fp8": float(self.runtime_dtype == getattr(torch, "float8_e4m3fn", torch.float16)),
            f"{self.label}.use_pinned": float(self.cfg.use_pinned_stage),
            f"{self.label}.use_async": float(self.cfg.use_async_stream),
        }

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is None:
            raise RuntimeError("benchmark_fn() must be called before verification")
        return self.output

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {
            "batch_size": self.cfg.batch_size,
            "num_heads": self.cfg.num_heads,
            "head_dim": self.cfg.head_dim,
            "max_seq_len": self.cfg.max_seq_len,
            "page_tokens": self.cfg.page_tokens,
            "decode_tokens": self.cfg.decode_tokens,
            "use_pinned_stage": self.cfg.use_pinned_stage,
            "use_async_stream": self.cfg.use_async_stream,
            "use_memmap": self.cfg.use_memmap,
            "shapes": {
                "host_cache": (
                    2,
                    self.cfg.batch_size,
                    self.cfg.num_heads,
                    self.cfg.max_seq_len,
                    self.cfg.head_dim,
                ),
                "hot_k": (
                    self.cfg.batch_size,
                    self.cfg.num_heads,
                    self.cfg.page_tokens,
                    self.cfg.head_dim,
                ),
                "hot_v": (
                    self.cfg.batch_size,
                    self.cfg.num_heads,
                    self.cfg.page_tokens,
                    self.cfg.head_dim,
                ),
            },
            "dtypes": {"hot_cache": str(self.runtime_dtype)},
        }

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)
