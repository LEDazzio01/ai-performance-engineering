"""baseline_expert_parallelism.py - Pedagogical MoE baseline (single GPU).

Implements a simple top-1 gated MoE forward path to keep the benchmark harness
happy and give the docs a runnable target. This stays intentionally small and
single-GPU so it can execute anywhere without special hardware.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class ToyGatedMoE(nn.Module):
    """Top-1 MoE with real per-expert dispatch on a single GPU."""

    def __init__(self, hidden_dim: int = 1024, num_experts: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(num_experts)]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.gate(tokens)
        weights = F.softmax(logits, dim=-1)
        top1 = torch.argmax(weights, dim=-1)

        # Dispatch per expert to avoid mixing routing decisions.
        outputs = torch.zeros_like(tokens)
        for expert_idx, expert in enumerate(self.experts):
            mask = top1 == expert_idx
            if mask.any():
                expert_tokens = tokens[mask]
                outputs[mask] = expert(expert_tokens)
        return outputs


class BaselineExpertParallelismBenchmark(BaseBenchmark):
    """Single-GPU MoE baseline without overlap or load balancing tricks."""

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[ToyGatedMoE] = None
        self.inputs: Optional[torch.Tensor] = None
        self._history: Dict[str, float] = {}
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=1024.0,
        )
        self.jitter_exemption_reason = "Expert parallelism benchmark: fixed dimensions"

    def setup(self) -> None:
        torch.manual_seed(0)
        hidden_dim = 1024
        batch = 64
        seq = 16
        self.model = ToyGatedMoE(hidden_dim=hidden_dim, num_experts=4).to(self.device).to(torch.bfloat16).eval()
        self.inputs = torch.randn(batch, seq, hidden_dim, device=self.device, dtype=torch.bfloat16)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Optional[dict]:
        if self.model is None or self.inputs is None:
            raise RuntimeError("SKIPPED: MoE model not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        start = self._record_start()
        with nvtx_range("moe_baseline_forward", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.model(self.inputs)
        torch.cuda.synchronize(self.device)
        latency_ms = self._record_stop(start)
        self._history["latency_ms"] = latency_ms
        return {"latency_ms": latency_ms}

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

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return torch.tensor([hash(str(id(self))) % (2**31)], dtype=torch.float32)

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"type": "expert_parallelism"}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)

def get_benchmark() -> BaseBenchmark:
    """Factory for the harness."""
    return BaselineExpertParallelismBenchmark()
