#!/usr/bin/env python3
"""Baseline: Disaggregated prefill/decode without optimization.

Basic disaggregated serving with separate pools but no optimizations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path
import time

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


class BaselineDisaggregatedPrefillDecode:
    """Baseline disaggregated serving."""
    
    def __init__(
        self,
        num_prefill_gpus: int = 2,
        num_decode_gpus: int = 6,
        batch_size: int = 8,
        prefill_length: int = 1024,
        decode_length: int = 128,
    ):
        self.num_prefill_gpus = num_prefill_gpus
        self.num_decode_gpus = num_decode_gpus
        self.batch_size = batch_size
        self.prefill_length = prefill_length
        self.decode_length = decode_length
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Disaggregated Serving")
        logger.info(f"  Prefill GPUs: {num_prefill_gpus}")
        logger.info(f"  Decode GPUs: {num_decode_gpus}")
    
    def setup(self):
        """Initialize models (simulated for both pools)."""
        hidden_size = 4096
        
        # Simulated prefill model
        self.prefill_model = nn.Linear(hidden_size, hidden_size).to(self.device)
        
        # Simulated decode model
        self.decode_model = nn.Linear(hidden_size, hidden_size).to(self.device)
        
        # Create inputs
        self.prefill_input = torch.randn(
            self.batch_size, self.prefill_length, hidden_size,
            device=self.device, dtype=torch.bfloat16
        )
        
        logger.info("Models initialized (baseline)")
    
    def run(self) -> Dict[str, float]:
        """Execute baseline disaggregated serving."""
        torch.cuda.synchronize()
        start_total = time.perf_counter()
        
        # Prefill phase (on prefill GPUs)
        prefill_start = time.perf_counter()
        prefill_output = self.prefill_model(self.prefill_input)
        kv_cache = prefill_output  # Simplified KV
        torch.cuda.synchronize()
        prefill_time = time.perf_counter() - prefill_start
        
        # Baseline: Blocking transfer of KV cache to decode pool
        # (No overlap, no compression)
        transfer_start = time.perf_counter()
        kv_cache_cpu = kv_cache.cpu()  # Transfer through CPU (baseline)
        kv_cache_decode = kv_cache_cpu.to(self.device)  # Back to GPU
        torch.cuda.synchronize()
        transfer_time = time.perf_counter() - transfer_start
        
        # Decode phase (on decode GPUs)
        decode_start = time.perf_counter()
        
        decode_outputs = []
        for _ in range(self.decode_length):
            # Simplified decode step
            decode_output = self.decode_model(kv_cache_decode[:, -1:, :])
            decode_outputs.append(decode_output)
        
        torch.cuda.synchronize()
        decode_time = time.perf_counter() - decode_start
        
        total_time = time.perf_counter() - start_total
        
        logger.info(f"Prefill: {prefill_time*1000:.2f} ms")
        logger.info(f"KV Transfer: {transfer_time*1000:.2f} ms")
        logger.info(f"Decode: {decode_time*1000:.2f} ms")
        logger.info(f"Total: {total_time*1000:.2f} ms")
        
        return {
            "total_latency_ms": total_time * 1000,
            "prefill_ms": prefill_time * 1000,
            "transfer_ms": transfer_time * 1000,
            "decode_ms": decode_time * 1000,
            "transfer_overhead_pct": (transfer_time / total_time) * 100,
        }
    
    def cleanup(self):
        """Clean up."""
        del self.prefill_model, self.decode_model, self.prefill_input
        torch.cuda.empty_cache()


def run_benchmark(
    num_prefill_gpus: int = 2,
    num_decode_gpus: int = 6,
    batch_size: int = 8,
    prefill_length: int = 1024,
    decode_length: int = 128,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run baseline disaggregated benchmark."""
    
    benchmark = BaselineDisaggregatedPrefillDecode(
        num_prefill_gpus=num_prefill_gpus,
        num_decode_gpus=num_decode_gpus,
        batch_size=batch_size,
        prefill_length=prefill_length,
        decode_length=decode_length,
    )
    benchmark.setup()
    
    config = BenchmarkConfig(iterations=3, warmup=5, profile_mode=profile)
    harness = BenchmarkHarness(mode=BenchmarkMode.INFERENCE, config=config)
    
    result = harness.benchmark(benchmark.run, name="baseline_disaggregated")
    
    metrics = benchmark.run()
    benchmark.cleanup()
    
    return {"mean_time_ms": result.timing.mean_ms, **metrics}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline Disaggregated Serving")
    parser.add_argument("--prefill-gpus", type=int, default=2)
    parser.add_argument("--decode-gpus", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prefill-length", type=int, default=1024)
    parser.add_argument("--decode-length", type=int, default=128)
    parser.add_argument("--profile", type=str, default="none")
    
    args = parser.parse_args()
    
    result = run_benchmark(
        num_prefill_gpus=args.prefill_gpus,
        num_decode_gpus=args.decode_gpus,
        batch_size=args.batch_size,
        prefill_length=args.prefill_length,
        decode_length=args.decode_length,
        profile=args.profile,
    )
    
    print(f"\n{'='*60}")
    print(f"Baseline Disaggregated Serving Results")
    print(f"{'='*60}")
    print(f"Prefill: {result['prefill_ms']:.2f} ms")
    print(f"Transfer: {result['transfer_ms']:.2f} ms ({result['transfer_overhead_pct']:.1f}%)")
    print(f"Decode: {result['decode_ms']:.2f} ms")
    print(f"Total: {result['total_latency_ms']:.2f} ms")
    print(f"{'='*60}\n")
    print(f"NOTE: Transfer overhead can be eliminated with NVLink pooling")


#============================================================================
# Benchmark Harness Integration
#============================================================================

class DisaggregatedPrefillDecodeBenchmark(BaseBenchmark):
    """Benchmark harness wrapper for baseline disaggregated serving."""

    def __init__(self):
        super().__init__()
        self.serving = None
        self.batch_size = 8
        self.prefill_length = 1024
        self.decode_length = 128
        self._last = 0.0
        
        tokens = self.batch_size * (self.prefill_length + self.decode_length)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Setup: Initialize baseline disaggregated serving."""
        torch.manual_seed(42)
        self.serving = BaselineDisaggregatedPrefillDecode(
            num_prefill_gpus=2,
            num_decode_gpus=6,
            batch_size=self.batch_size,
            prefill_length=self.prefill_length,
            decode_length=self.decode_length,
        )
        self.serving.setup()

    def benchmark_fn(self) -> None:
        """Benchmark: Baseline prefill-decode."""
        if self.serving is not None:
            result = self.serving.run()
            self._last = result.get("total_latency_ms", 0.0) if isinstance(result, dict) else result
        self._synchronize()

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        if self.serving is not None:
            self.serving.cleanup()
            self.serving = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)
    
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
        if self.serving is None:
            return "Disaggregated serving not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return DisaggregatedPrefillDecodeBenchmark()

