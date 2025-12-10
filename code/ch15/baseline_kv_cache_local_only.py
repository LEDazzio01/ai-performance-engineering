"""baseline_kv_cache_local_only.py

Baseline KV-cache strategy: keep everything in local HBM and evict to host when full.
No NVLink pooling or peer placement is used.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineKVCacheLocalOnlyBenchmark(BaseBenchmark):
    """Local-only KV cache with host spill."""

    def __init__(self):
        super().__init__()
        self.jitter_exemption_reason = "Benchmark: fixed dimensions"
        self.model: Optional[nn.MultiheadAttention] = None
        self.cache: Optional[torch.Tensor] = None
        self.hidden = 256
        self.heads = 8
        self.batch = 4
        self.seq_len = 64
        self.cache_limit = 128  # tokens before spill
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        self.model = nn.MultiheadAttention(self.hidden, self.heads, batch_first=True).to(self.device).eval()
        # KV cache stored locally only
        self.cache = torch.zeros(self.cache_limit, self.batch, self.hidden, device=self.device)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.model is not None and self.cache is not None
        with self._nvtx_range("baseline_kv_cache_local_only"):
            keys: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            for _ in range(self.seq_len):
                q = torch.randn(self.batch, 1, self.hidden, device=self.device)
                k = torch.randn(self.batch, 1, self.hidden, device=self.device)
                v = torch.randn(self.batch, 1, self.hidden, device=self.device)
                keys.append(k)
                values.append(v)

                if len(keys) > self.cache_limit:
                    # Spill oldest to host (slow)
                    keys.pop(0).cpu()
                    values.pop(0).cpu()

                k_all = torch.cat(keys, dim=1)
                v_all = torch.cat(values, dim=1)
                out, _ = self.model(q, k_all, v_all)
                _ = out.sum()
            self._synchronize()

    def teardown(self) -> None:
        self.model = None
        self.cache = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_inference_metrics
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


def get_benchmark() -> BaseBenchmark:
    return BaselineKVCacheLocalOnlyBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(BaselineKVCacheLocalOnlyBenchmark)
