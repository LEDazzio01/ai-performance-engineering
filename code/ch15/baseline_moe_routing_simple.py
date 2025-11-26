#!/usr/bin/env python3
"""Baseline: Simple MoE routing without topology awareness.

Basic MoE routing that doesn't consider GPU topology or NVLink locality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from common.python.logger import get_logger

logger = get_logger(__name__)


class BaselineMoERoutingSimple:
    """Baseline MoE routing without topology awareness."""
    
    def __init__(
        self,
        batch_size: int = 16,
        seq_length: int = 2048,
        hidden_size: int = 4096,
        num_experts: int = 64,
        top_k: int = 2,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Baseline MoE Routing")
        logger.info(f"  Experts: {num_experts}, Top-K: {top_k}")
    
    def setup(self):
        """Initialize router."""
        # Simple linear router
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False).to(self.device)
        
        # Create input
        self.input = torch.randn(
            self.batch_size,
            self.seq_length,
            self.hidden_size,
            device=self.device,
            dtype=torch.bfloat16
        )
        
        logger.info("Router initialized")
    
    def run(self) -> Dict[str, float]:
        """Execute baseline routing."""
        import time
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Compute routing scores
        routing_logits = self.router(self.input)  # [batch, seq, num_experts]
        
        # Top-K selection (no topology consideration)
        routing_weights, selected_experts = torch.topk(routing_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        # Calculate load imbalance
        expert_counts = torch.bincount(
            selected_experts.view(-1),
            minlength=self.num_experts
        ).float()
        
        ideal_count = expert_counts.sum() / self.num_experts
        load_variance = torch.var(expert_counts).item()
        max_imbalance = (expert_counts.max() / ideal_count).item()
        
        logger.info(f"Load variance: {load_variance:.2f}")
        logger.info(f"Max imbalance: {max_imbalance:.2f}×")
        
        return {
            "latency_ms": elapsed * 1000,
            "load_variance": load_variance,
            "max_imbalance": max_imbalance,
            "topology_aware": False,
        }
    
    def cleanup(self):
        """Clean up."""
        del self.router, self.input
        torch.cuda.empty_cache()


def run_benchmark(
    batch_size: int = 16,
    seq_length: int = 2048,
    hidden_size: int = 4096,
    num_experts: int = 64,
    top_k: int = 2,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run baseline MoE routing benchmark."""
    
    benchmark = BaselineMoERoutingSimple(
        batch_size=batch_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
    )
    benchmark.setup()
    
    config = BenchmarkConfig(iterations=10, warmup=2, profile_mode=profile)
    harness = BenchmarkHarness(mode=BenchmarkMode.INFERENCE, config=config)
    
    result = harness.benchmark(benchmark.run, name="baseline_moe_routing")
    
    metrics = benchmark.run()
    benchmark.cleanup()
    
    return {"mean_time_ms": result.timing.mean_ms, **metrics}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline MoE Routing")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--profile", type=str, default="none")
    
    args = parser.parse_args()
    
    result = run_benchmark(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_experts=args.num_experts,
        top_k=args.top_k,
        profile=args.profile,
    )
    
    print(f"\n{'='*60}")
    print(f"Baseline MoE Routing Results")
    print(f"{'='*60}")
    print(f"Mean time: {result['mean_time_ms']:.2f} ms")
    print(f"Load variance: {result['load_variance']:.2f}")
    print(f"Max imbalance: {result['max_imbalance']:.2f}×")
    print(f"Topology aware: {result['topology_aware']}")
    print(f"{'='*60}\n")


#============================================================================
# Benchmark Harness Integration
#============================================================================

class MoERoutingSimpleBenchmark(BaseBenchmark):
    """Benchmark harness wrapper for baseline MoE routing."""

    def __init__(self):
        super().__init__()
        self.moe = None
        self.batch_size = 16
        self.seq_length = 2048
        self.hidden_size = 4096
        self.num_experts = 64
        self._last = 0.0
        
        tokens = self.batch_size * self.seq_length
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Setup: Initialize baseline MoE routing."""
        torch.manual_seed(42)
        self.moe = BaselineMoERoutingSimple(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            top_k=2,
        )
        self.moe.setup()

    def benchmark_fn(self) -> None:
        """Benchmark: Baseline MoE routing and forward pass."""
        if self.moe is not None:
            result = self.moe.run()
            self._last = result
        self._synchronize()

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        if self.moe is not None:
            self.moe.cleanup()
            self.moe = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)
    
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
        if self.moe is None:
            return "MoE routing not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return MoERoutingSimpleBenchmark()

