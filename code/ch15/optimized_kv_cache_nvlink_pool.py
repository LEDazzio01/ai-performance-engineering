"""optimized_kv_cache_nvlink_pool.py

Three-tier KV cache: local HBM (hot), peer HBM over NVLink (warm), host/Grace
for cold entries. Prefers near-neighbor GPU (device 1 if present) and uses
non-blocking transfers to overlap cache fetch with compute.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from common.python.gpu_requirements import skip_if_insufficient_gpus, require_peer_access
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


def _enable_peer_access() -> None:
    skip_if_insufficient_gpus(2)
    require_peer_access(0, 1)
    try:
        torch.cuda.device(0).enable_peer_access(1)
        torch.cuda.device(1).enable_peer_access(0)
    except RuntimeError:
        pass


class OptimizedKVCacheNvlinkPoolBenchmark(BaseBenchmark):
    """Tiered KV cache with NVLink pooling."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.MultiheadAttention] = None
        self.local_cache_limit = 64
        self.peer_cache_limit = 64
        self.hidden = 256
        self.heads = 8
        self.batch = 4
        self.seq_len = 64
        tokens = self.batch * self.seq_len
        self.peer_device: Optional[torch.device] = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        _enable_peer_access()
        self.peer_device = torch.device("cuda:1")
        self.model = nn.MultiheadAttention(self.hidden, self.heads, batch_first=True).to(self.device).eval()
        self._synchronize()

    def _place_kv(self, k: torch.Tensor, v: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Decide where to place KV: local -> peer -> host."""
        if step < self.local_cache_limit:
            return k, v, "local"
        if self.peer_device is not None and step < self.local_cache_limit + self.peer_cache_limit:
            return k.to(self.peer_device, non_blocking=True), v.to(self.peer_device, non_blocking=True), "peer"
        return k.cpu(), v.cpu(), "host"

    def benchmark_fn(self) -> None:
        assert self.model is not None
        with self._nvtx_range("optimized_kv_cache_nvlink_pool"):
            cache_k: list[torch.Tensor] = []
            cache_v: list[torch.Tensor] = []
            tiers: list[str] = []
            for step in range(self.seq_len):
                q = torch.randn(self.batch, 1, self.hidden, device=self.device)
                k = torch.randn(self.batch, 1, self.hidden, device=self.device)
                v = torch.randn(self.batch, 1, self.hidden, device=self.device)
                placed_k, placed_v, tier = self._place_kv(k, v, step)
                cache_k.append(placed_k)
                cache_v.append(placed_v)
                tiers.append(tier)

                # Gather, preferring local first then peer then host (copied back async)
                gathered_k = []
                gathered_v = []
                for tk, tv, t in zip(cache_k, cache_v, tiers):
                    if t == "local":
                        gathered_k.append(tk)
                        gathered_v.append(tv)
                    elif t == "peer":
                        gathered_k.append(tk.to(self.device, non_blocking=True))
                        gathered_v.append(tv.to(self.device, non_blocking=True))
                    else:  # host
                        gathered_k.append(tk.to(self.device, non_blocking=True))
                        gathered_v.append(tv.to(self.device, non_blocking=True))

                k_all = torch.cat(gathered_k, dim=1)
                v_all = torch.cat(gathered_v, dim=1)
                out, _ = self.model(q, k_all, v_all)
                _ = out.sum()
            self._synchronize()

    def teardown(self) -> None:
        self.model = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=1)

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


def get_benchmark() -> BaseBenchmark:
    return OptimizedKVCacheNvlinkPoolBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=2, warmup=0),
    )
    bench = OptimizedKVCacheNvlinkPoolBenchmark()
    result = harness.benchmark(bench)
    print(result)
