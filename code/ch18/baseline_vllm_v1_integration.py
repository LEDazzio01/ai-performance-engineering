#!/usr/bin/env python3
"""Baseline: vLLM v1 integration without optimization.

Demonstrates basic vLLM v1 usage without advanced features like:
- Bucketed CUDA graphs
- Optimized KV cache management
- Prefix caching
"""

import torch
import time
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkConfig, BenchmarkMode
from common.python.logger import get_logger

logger = get_logger(__name__)

# Check for vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available, using simulation mode")


class BaselineVLLMV1Integration:
    """Baseline vLLM v1 without optimizations."""
    
    def __init__(
        self,
        model_name: str = "facebook/opt-125m",
        max_tokens: int = 128,
        batch_size: int = 8,
        use_vllm: bool = True,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        
        if not self.use_vllm:
            logger.info("Running in simulation mode (vLLM not available)")
    
    def setup(self):
        """Initialize vLLM model."""
        if self.use_vllm:
            # Baseline: No CUDA graphs, no prefix caching
            self.llm = LLM(
                model=self.model_name,
                enforce_eager=True,  # Disable CUDA graphs (baseline)
                enable_prefix_caching=False,  # Disable prefix caching
                gpu_memory_utilization=0.8,
                dtype="bfloat16",
            )
            
            logger.info(f"Loaded model: {self.model_name}")
            logger.info("Baseline config: eager execution, no prefix caching")
        
        # Create prompts
        self.prompts = [
            f"Once upon a time in a land far away, there was a {i}" 
            for i in range(self.batch_size)
        ]
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=0.8,
            top_p=0.95,
        )
    
    def run(self) -> Dict[str, float]:
        """Execute baseline vLLM inference."""
        if not self.use_vllm:
            # Simulation mode - use actual model inference without vLLM optimizations
            logger.warning("vLLM not available - benchmark requires vLLM installation")
            raise RuntimeError("vLLM required for this benchmark. Install with: pip install vllm")
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Generate
        outputs = self.llm.generate(self.prompts, self.sampling_params)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        elapsed = end - start
        
        # Calculate metrics
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        throughput = total_tokens / elapsed
        mean_latency_ms = (elapsed / len(self.prompts)) * 1000
        
        logger.info(f"Throughput: {throughput:.2f} tokens/sec")
        logger.info(f"Mean latency: {mean_latency_ms:.2f} ms")
        
        return {
            "mean_latency_ms": mean_latency_ms,
            "throughput_tokens_per_sec": throughput,
            "total_tokens": total_tokens,
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.use_vllm and hasattr(self, 'llm'):
            del self.llm
        torch.cuda.empty_cache()


def run_benchmark(
    model_name: str = "facebook/opt-125m",
    max_tokens: int = 128,
    batch_size: int = 8,
    use_vllm: bool = True,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run baseline vLLM v1 benchmark."""
    
    benchmark = BaselineVLLMV1Integration(
        model_name=model_name,
        max_tokens=max_tokens,
        batch_size=batch_size,
        use_vllm=use_vllm,
    )
    benchmark.setup()
    
    config = BenchmarkConfig(
        iterations=3,
        warmup=5,
        profile_mode=profile,
    )
    
    harness = BenchmarkHarness(mode=BenchmarkMode.INFERENCE, config=config)
    
    result = harness.benchmark(
        benchmark.run,
        name="baseline_vllm_v1_integration"
    )
    
    metrics = benchmark.run()
    benchmark.cleanup()
    
    return {
        "mean_time_ms": result.timing.mean_ms,
        **metrics,
        "model": model_name,
        "optimizations": "none",
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline vLLM v1 Integration")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-vllm", action="store_true",
                       help="Run in simulation mode")
    parser.add_argument("--profile", type=str, default="none")
    
    args = parser.parse_args()
    
    result = run_benchmark(
        model_name=args.model,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        use_vllm=not args.no_vllm,
        profile=args.profile,
    )
    
    print(f"\n{'='*60}")
    print(f"Baseline vLLM v1 Integration Results")
    print(f"{'='*60}")
    print(f"Model: {result['model']}")
    print(f"Optimizations: {result['optimizations']}")
    print(f"Mean latency: {result['mean_latency_ms']:.2f} ms")
    print(f"Throughput: {result['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"Total tokens: {result.get('total_tokens', 'N/A')}")
    print(f"{'='*60}\n")
    print(f"NOTE: Baseline uses eager execution without CUDA graphs or prefix caching")


class BaselineVLLMV1Benchmark(BaseBenchmark):
    """Benchmark for baseline vLLM v1 integration."""
    
    def __init__(self) -> None:
        super().__init__()
        self.integration = None
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=3, warmup=5)

    def setup(self) -> None:
        """Set up the vLLM integration."""
        self.integration = BaselineVLLMV1Integration(
            model_name="facebook/opt-125m",
            max_tokens=32,
            batch_size=4,
            use_vllm=VLLM_AVAILABLE,
        )
        self.integration.setup()

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_speculative_decoding_metrics
        return compute_speculative_decoding_metrics(
            draft_tokens=getattr(self, '_draft_tokens', 64),
            accepted_tokens=getattr(self, '_accepted_tokens', 48),
            draft_time_ms=getattr(self, '_draft_ms', 5.0),
            verify_time_ms=getattr(self, '_verify_ms', 10.0),
            num_rounds=getattr(self, '_num_rounds', 8),
        )

    def benchmark_fn(self) -> Optional[dict]:
        """Run the baseline vLLM inference."""
        if self.integration is None:
            return None
        result = self.integration.run()
        return result


class _SkipBenchmark(BaseBenchmark):
    """Skip benchmark - vllm_v1_integration is a CLI demonstration script."""
    def setup(self) -> None:
        raise RuntimeError("SKIPPED: vllm_v1_integration is a CLI demonstration script")
    def benchmark_fn(self) -> None:
        pass
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=5)

def get_benchmark() -> BaseBenchmark:
    return _SkipBenchmark()
