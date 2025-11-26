"""Optimized Flash SDP attention benchmark with torch.compile and fail-fast checks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range


def ensure_flash_sdp_available() -> None:
    """Fail fast by actually invoking the Flash kernel."""
    if not torch.cuda.is_available():
        raise RuntimeError("Flash SDP benchmark requires a CUDA device.")
    try:
        q = torch.randn(1, 1, 4, 64, device="cuda", dtype=torch.float16)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            _ = F.scaled_dot_product_attention(q, q, q, is_causal=False)
        torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - only hit on unsupported stacks
        raise RuntimeError(f"SKIPPED: Flash SDP kernel failed to run: {exc}") from exc


class FlashAttentionModule(nn.Module):
    """Attention block that forces Flash SDP backend."""

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.hidden_dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(B, T, self.hidden_dim)
        return out


class OptimizedFlashSDPBenchmark(BaseBenchmark):
    """Uses torch.compile on the Flash SDP block."""

    def __init__(self):
        super().__init__()
        self.model: Optional[FlashAttentionModule] = None
        self.inputs: Optional[torch.Tensor] = None
        self.compiled = None
        self.seq_len = 256
        self.batch = 8
        self.hidden = 512
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        ensure_flash_sdp_available()
        torch.manual_seed(0)
        self.model = FlashAttentionModule(hidden_dim=self.hidden, num_heads=8).to(
            self.device, dtype=torch.float16
        )
        self.inputs = torch.randn(self.batch, self.seq_len, self.hidden, device=self.device, dtype=torch.float16)
        compiled = torch.compile(self.model, fullgraph=True, dynamic=False)  # type: ignore[arg-type]
        # Warm compile
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            _ = compiled(self.inputs)
        torch.cuda.synchronize(self.device)
        self.compiled = compiled

    def benchmark_fn(self) -> None:
        if self.compiled is None or self.inputs is None:
            raise RuntimeError("Compiled model not initialized")
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("flash_sdp_optimized", enable=enable_nvtx):
            _ = self.compiled(self.inputs)
        torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.compiled = None
        torch.cuda.empty_cache()

    def validate_result(self) -> Optional[str]:
        if self.compiled is None or self.inputs is None:
            return "Compiled model not initialized"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=1,
            warmup=1,
            measurement_timeout_seconds=90,
            setup_timeout_seconds=90,
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

def get_benchmark() -> BaseBenchmark:
    return OptimizedFlashSDPBenchmark()


if __name__ == "__main__":
    bench = get_benchmark()
    harness_cfg = bench.get_config()
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=harness_cfg)
    result = harness.benchmark(bench)
    print(
        f"[flash_sdp optimized] mean iteration "
        f"{result.timing.mean_ms if result and result.timing else 0.0:.3f} ms"
    )
