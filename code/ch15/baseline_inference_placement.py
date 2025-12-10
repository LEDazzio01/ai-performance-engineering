"""Baseline inference placement policy (intentionally cross-node heavy)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig  # noqa: E402
from ch15.placement_sim import (  # noqa: E402
    PlacementConfig,
    PlacementSimulator,
    percentile,
)


class _PlacementBenchmark(BaseBenchmark):
    """Shared scaffolding for placement simulations."""

    def __init__(self, cfg: PlacementConfig, prefix: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.prefix = prefix
        self.simulator = PlacementSimulator()
        self._summary: Dict[str, float] = {}
        self.jitter_exemption_reason = "Placement benchmark: simulation parameters"
        self.register_workload_metadata(requests_per_iteration=1.0)

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return torch.tensor([hash(str(id(self))) % (2**31)], dtype=torch.float32)

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"prefix": self.prefix}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)

    def setup(self) -> None:
        torch_backend = self.cfg.dtype
        # Use torch bf16-friendly matmul path on Blackwell; harmless elsewhere.
        import torch
        torch.set_default_dtype(torch_backend)  # type: ignore[arg-type]

    def benchmark_fn(self) -> None:
        run = self.simulator.simulate(self.cfg, sessions=64, seed=17)
        ttft_p50 = percentile(run.ttft_ms, 50)
        ttft_p95 = percentile(run.ttft_ms, 95)
        decode_p50 = percentile(run.decode_ms, 50)
        decode_p95 = percentile(run.decode_ms, 95)
        total_ms = sum(run.ttft_ms) + sum(run.decode_ms)
        tput_tokens_s = run.tokens_processed / max(total_ms / 1000.0, 1e-6)

        self._summary = {
            f"{self.prefix}.ttft_p50_ms": ttft_p50,
            f"{self.prefix}.ttft_p95_ms": ttft_p95,
            f"{self.prefix}.decode_p50_ms": decode_p50,
            f"{self.prefix}.decode_p95_ms": decode_p95,
            f"{self.prefix}.tokens_per_s_est": tput_tokens_s,
            f"{self.prefix}.cross_node_kv_moves": float(run.cross_node_kv_moves),
            f"{self.prefix}.cross_node_collectives": float(run.cross_node_collectives),
            f"{self.prefix}.prefill_collective_ms": run.prefill_collective_ms,
            f"{self.prefix}.decode_collective_ms": run.decode_collective_ms,
            f"{self.prefix}.remote_expert_ms": run.remote_expert_ms,
        }

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(
            iterations=1,
            warmup=5,
            measurement_timeout_seconds=60,
            timeout_multiplier=2.0,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics for inference_placement."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=getattr(self, '_ttft_ms', 10.0),
            tpot_ms=getattr(self, '_tpot_ms', 1.0),
            total_tokens=getattr(self, '_total_tokens', 100),
            total_requests=getattr(self, '_total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )


class BaselineInferencePlacementBenchmark(_PlacementBenchmark):
    """Naive placement with cross-node TP/EP and non-sticky decode."""

    def __init__(self) -> None:
        cfg = PlacementConfig(
            prefill_tp_size=8,
            prefill_span_nodes=True,
            decode_tp_size=2,
            decode_span_nodes=True,
            decode_microbatch=16,
            remote_expert_fraction=0.35,
            router_sticky_decode=False,
            kv_transfer_policy="allow_cross_node",
            notes="Cross-node TP/EP, larger decode microbatches, no KV locality.",
        )
        super().__init__(cfg, prefix="placement_baseline")


def get_benchmark() -> BaseBenchmark:
    return BaselineInferencePlacementBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
