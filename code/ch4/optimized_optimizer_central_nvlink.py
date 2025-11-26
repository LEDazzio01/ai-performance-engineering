"""optimized_optimizer_central_nvlink.py

Centralized optimizer state on a single GPU (typically within the same NVSwitch
island) with peer access enabled. Remote GPUs ship gradients to the central
GPU over NVLink; updated weights are multicast back.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from common.python.gpu_requirements import skip_if_insufficient_gpus, require_peer_access
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedOptimizerCentralNvlinkBenchmark(BaseBenchmark):
    """Centralized optimizer state (one shard per switch island)."""

    def __init__(self):
        super().__init__()
        self.models: List[nn.Linear] = []
        self.master_weights: List[torch.Tensor] = []
        self.momentum: List[torch.Tensor] = []
        self.batch_size = 8
        self.hidden = 512
        self.root_device = torch.device("cuda:0")
        tokens = self.batch_size * self.hidden
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def _enable_peer_access(self) -> None:
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
                        # Already enabled or unsupported; ignore
                        pass

    def setup(self) -> None:
        torch.manual_seed(123)
        self._enable_peer_access()
        num_gpus = max(1, torch.cuda.device_count())
        skip_if_insufficient_gpus(2)

        for rank in range(num_gpus):
            device = f"cuda:{rank}"
            model = nn.Linear(self.hidden, self.hidden).to(device)
            self.models.append(model)
            # Master copies live on the root device
            master_w = model.weight.detach().to(self.root_device)
            master_m = torch.zeros_like(master_w, dtype=torch.float32, device=self.root_device)
            self.master_weights.append(master_w)
            self.momentum.append(master_m)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert len(self.models) == len(self.master_weights) == len(self.momentum)
        with self._nvtx_range("optimized_optimizer_central_nvlink"):
            for model, master_w, mom in zip(self.models, self.master_weights, self.momentum):
                x = torch.randn(self.batch_size, self.hidden, device=model.weight.device)
                y = model(x)
                loss = y.pow(2).mean()
                loss.backward()

                # Ship gradient to root over NVLink (non-blocking if available)
                grad_root = model.weight.grad.to(self.root_device, non_blocking=True)
                with torch.no_grad():
                    mom.mul_(0.9).add_(grad_root)
                    master_w.add_(-1e-3, mom)
                # Multicast updated weights back
                model.weight.data.copy_(master_w.to(model.weight.device, non_blocking=True))
                model.bias.grad.zero_()
                model.weight.grad.zero_()
            self._synchronize()

    def teardown(self) -> None:
        self.models.clear()
        self.master_weights.clear()
        self.momentum.clear()
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=2)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        if not self.models or not self.master_weights:
            return "Models or optimizer state not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedOptimizerCentralNvlinkBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=4, warmup=1),
    )
    bench = OptimizedOptimizerCentralNvlinkBenchmark()
    result = harness.benchmark(bench)
    print(result)
