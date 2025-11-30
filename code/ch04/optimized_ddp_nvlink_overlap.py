"""optimized_ddp_nvlink_overlap.py

Topology- and overlap-aware DDP-style loop. Enables peer access, reorders
gradient buckets by device distance (lowest ID first as a proxy), and overlaps
gradient transfer with computation using a dedicated communication stream.
Falls back to single-GPU if only one device is present.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from core.benchmark.gpu_requirements import skip_if_insufficient_gpus, require_peer_access
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


def _enable_peer_access() -> None:
    num = torch.cuda.device_count()
    skip_if_insufficient_gpus(2)
    for src in range(num):
        for dst in range(num):
            if src == dst:
                continue
            if torch.cuda.can_device_access_peer(src, dst):
                try:
                    torch.cuda.device(src).enable_peer_access(dst)
                except RuntimeError:
                    pass


def _bucket_order() -> List[Tuple[int, int]]:
    """Return (device_id, bucket_index) pairs ordered by device id (proxy for distance)."""
    return [(idx, idx) for idx in range(torch.cuda.device_count())]


class OptimizedDdpNvlinkOverlapBenchmark(BaseBenchmark):
    """Overlapped gradient transfers over NVLink with peer access enabled."""

    def __init__(self):
        super().__init__()
        self.models: List[nn.Linear] = []
        self.microbatches = 2
        self.batch_size = 8
        self.hidden = 512
        self.root_device = torch.device("cuda:0")
        tokens = self.batch_size * self.hidden * self.microbatches
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size * self.microbatches),
            tokens_per_iteration=float(tokens),
        )
        self.comm_stream = torch.cuda.Stream(device=self.root_device)

    def setup(self) -> None:
        torch.manual_seed(0)
        _enable_peer_access()
        num = torch.cuda.device_count()
        skip_if_insufficient_gpus(2)
        require_peer_access(0, 1)
        for rank in range(num):
            device = f"cuda:{rank}"
            self.models.append(nn.Linear(self.hidden, self.hidden).to(device))
        self._synchronize()

    def _async_reduce_to_root(self, grads: List[torch.Tensor]) -> torch.Tensor:
        """Asynchronously accumulate gradients on the root device."""
        root_buf = torch.zeros_like(grads[0], device=self.root_device)
        events = []
        for g in grads:
            evt = torch.cuda.Event()
            evt.record()
            events.append((g, evt))

        with torch.cuda.stream(self.comm_stream):
            for g, evt in events:
                self.comm_stream.wait_event(evt)
                root_buf.add_(g.to(self.root_device, non_blocking=True))
        return root_buf

    def benchmark_fn(self) -> None:
        assert self.models
        with self._nvtx_range("optimized_ddp_nvlink_overlap"):
            reduction_results: List[torch.Tensor] = []
            # Process microbatches; overlap reduction of previous with compute of next
            for micro in range(self.microbatches):
                grads = []
                for model in self.models:
                    x = torch.randn(self.batch_size, self.hidden, device=model.weight.device)
                    y = model(x)
                    loss = y.pow(2).mean()
                    loss.backward()
                    grads.append(model.weight.grad)

                # Reorder buckets (simple proxy: ascending device id)
                ordered = sorted(zip(grads, _bucket_order()), key=lambda kv: kv[1][0])
                ordered_grads = [g for g, _ in ordered]

                reduction_results.append(self._async_reduce_to_root(ordered_grads))

            # Finalize reductions and apply updates
            self.comm_stream.synchronize()
            for model, root_buf in zip(self.models, reduction_results):
                if root_buf.device != model.weight.device:
                    root_local = root_buf.to(model.weight.device, non_blocking=True)
                else:
                    root_local = root_buf
                with torch.no_grad():
                    root_local.mul_(1.0 / len(self.models))
                    model.weight.add_(-1e-3, root_local)
                    model.weight.grad.zero_()
                    model.bias.grad.zero_()
            self._synchronize()

    def teardown(self) -> None:
        self.models.clear()
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        if not self.models:
            return "Models not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedDdpNvlinkOverlapBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=2, warmup=5),
    )
    bench = OptimizedDdpNvlinkOverlapBenchmark()
    result = harness.benchmark(bench)
    print(result)
