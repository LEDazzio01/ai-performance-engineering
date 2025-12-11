"""labs.moe_cuda/optimized_decode_attention.py - Optimized decode attention with BF16.

Optimization: Uses BF16 precision with nn.MultiheadAttention for faster computation.
Fair comparison: Same operations as baseline but with lower precision.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.utils.compile_utils import enable_tf32
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range


class OptimizedDecodeAttentionBenchmark(BaseBenchmark):
    """Optimized decode attention using BF16 precision.
    
    Optimization over baseline:
    - BF16 instead of FP32 (2x less memory bandwidth, faster on Tensor Cores)
    - TF32 enabled for matmuls
    - Same nn.MultiheadAttention for fair comparison
    """

    def __init__(self) -> None:
        super().__init__()
        # Realistic decode workload where BF16 optimization shows benefit
        self.batch = 32
        self.num_heads = 12
        self.kv_seq = 512
        self.head_dim = 64
        self.module: Optional[nn.Module] = None
        self.q: Optional[torch.Tensor] = None
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        tokens = self.batch * self.kv_seq
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self._history: Dict[str, List[float]] = {"latency_ms": []}

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("labs.moe_cuda decode attention requires CUDA")

        # Optimization: Enable TF32 for faster matmuls
        enable_tf32()
        torch.manual_seed(0)
        
        # Same module as baseline but with BF16
        self.module = nn.MultiheadAttention(
            embed_dim=self.num_heads * self.head_dim,
            num_heads=self.num_heads,
            batch_first=True,
            device=self.device,
            dtype=torch.bfloat16,  # Optimization: BF16 instead of FP32
        )
        
        # Same tensor shapes as baseline
        self.q = torch.randn(
            self.batch,
            1,
            self.num_heads * self.head_dim,
            device=self.device,
            dtype=torch.bfloat16,  # Optimization: BF16
        )
        self.k = torch.randn(
            self.batch,
            self.kv_seq,
            self.num_heads * self.head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.v = torch.randn_like(self.k)
        torch.cuda.synchronize(self.device)
        self.output = None

    def benchmark_fn(self) -> Dict[str, List[float]]:
        if any(t is None for t in (self.module, self.q, self.k, self.v)):
            raise RuntimeError("Decode tensors missing")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("moe_cuda_decode_optimized", enable=enable_nvtx):
            with torch.inference_mode():
                start = self._record_start()
                attn_out, _ = self.module(self.q, self.k, self.v)
                torch.cuda.synchronize(self.device)
                self._history["latency_ms"].append(self._record_stop(start))
                self.output = attn_out.detach().float().clone()
        return {"decode_ms": self._history["latency_ms"]}

    def teardown(self) -> None:
        self.module = None
        self.q = None
        self.k = None
        self.v = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=8, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if not self._history["latency_ms"]:
            return None
        return {
            "decode.mean_ms": float(sum(self._history["latency_ms"]) / len(self._history["latency_ms"]))
        }

    def validate_result(self) -> Optional[str]:
        if any(t is None for t in (self.module, self.q, self.k, self.v)):
            return "Decode tensors missing"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is None:
            raise RuntimeError("benchmark_fn() must be called before verification")
        return self.output

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {
            "batch": self.batch,
            "kv_seq": self.kv_seq,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "shapes": {
                "q": (self.batch, 1, self.num_heads * self.head_dim),
                "k": (self.batch, self.kv_seq, self.num_heads * self.head_dim),
                "v": (self.batch, self.kv_seq, self.num_heads * self.head_dim),
                "out": (self.batch, 1, self.num_heads * self.head_dim),
            },
            "dtypes": {"q": "bfloat16"},
        }

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    return OptimizedDecodeAttentionBenchmark()
