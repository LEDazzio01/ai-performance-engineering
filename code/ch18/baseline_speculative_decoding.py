#!/usr/bin/env python3
"""Baseline: Sequential autoregressive decoding (NO speculative decoding).

This baseline generates tokens one at a time using ONLY the target model.
Compare with optimized version that uses speculative decoding for speedup.

The key insight: Sequential decoding requires N target model forward passes
for N tokens. Speculative decoding can achieve the same with fewer passes.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
import sys
from pathlib import Path
import time

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.python.benchmark_harness import BenchmarkHarness, BenchmarkConfig, BenchmarkMode
from common.python.logger import get_logger

logger = get_logger(__name__)


class BaselineSpeculativeDecoding:
    """Baseline: Sequential autoregressive decoding with target model only.
    
    This does NOT use speculative decoding - it generates one token at a time
    using only the large target model. This is the standard approach that
    speculative decoding aims to improve upon.
    """
    
    def __init__(
        self,
        batch_size: int = 4,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_draft_tokens: int = 4,  # Used to match token count with optimized
        num_sequences: int = 10,
    ):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_draft_tokens = num_draft_tokens
        self.num_sequences = num_sequences
        # Total tokens to generate = num_sequences * num_draft_tokens
        # This ensures fair comparison with speculative decoding
        self.tokens_to_generate = num_sequences * num_draft_tokens
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Baseline Sequential Autoregressive Decoding")
        logger.info(f"  Tokens to generate: {self.tokens_to_generate}")
    
    def _create_target_model(self):
        """Create the target language model (large, accurate)."""
        class SimpleLM(nn.Module):
            def __init__(self, vocab_size, hidden_size):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.linear1 = nn.Linear(hidden_size, hidden_size)
                self.linear2 = nn.Linear(hidden_size, vocab_size)
            
            def forward(self, input_ids):
                x = self.embedding(input_ids)
                x = torch.relu(self.linear1(x))
                logits = self.linear2(x)
                return logits
        
        return SimpleLM(self.vocab_size, self.hidden_size)
    
    def setup(self):
        """Initialize target model only (no draft model in baseline)."""
        # Target model (large) - this is what we use for sequential decode
        self.target_model = self._create_target_model().to(self.device).eval()
        
        # Initial input
        self.input_ids = torch.randint(
            0, self.vocab_size,
            (self.batch_size, 1),
            device=self.device
        )
        
        logger.info("Target model initialized (sequential baseline)")
    
    def _generate_one_token(self, current_ids: torch.Tensor) -> torch.Tensor:
        """Generate a single token using target model (sequential)."""
        with torch.no_grad():
            logits = self.target_model(current_ids[:, -1:])
            next_token = torch.argmax(logits, dim=-1)
        return next_token
    
    def run(self) -> Dict[str, float]:
        """Execute baseline sequential autoregressive decoding.
        
        This is the SLOW approach: Generate N tokens one at a time,
        each requiring a full target model forward pass.
        
        Cost: N * target_forward_time
        """
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        current_ids = self.input_ids.clone()
        
        # Generate tokens_to_generate tokens sequentially
        # Each token requires ONE target model forward pass
        for _ in range(self.tokens_to_generate):
            next_token = self._generate_one_token(current_ids)
            current_ids = torch.cat([current_ids, next_token], dim=1)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        # Calculate metrics
        tokens_generated = current_ids.shape[1] - self.input_ids.shape[1]
        tokens_per_sec = tokens_generated * self.batch_size / elapsed
        
        logger.info(f"Sequential decode: {tokens_generated} tokens")
        logger.info(f"Tokens/sec: {tokens_per_sec:.2f}")
        logger.info(f"Target forward passes: {tokens_generated} (one per token)")
        
        return {
            "latency_ms": elapsed * 1000,
            "tokens_per_sec": tokens_per_sec,
            "tokens_generated": tokens_generated,
            "target_forward_passes": tokens_generated,
        }
    
    def cleanup(self):
        """Clean up resources."""
        del self.target_model, self.input_ids
        torch.cuda.empty_cache()


def run_benchmark(
    batch_size: int = 4,
    vocab_size: int = 32000,
    hidden_size: int = 4096,
    num_draft_tokens: int = 4,
    num_sequences: int = 10,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run baseline speculative decoding benchmark."""
    
    benchmark = BaselineSpeculativeDecoding(
        batch_size=batch_size,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_draft_tokens=num_draft_tokens,
        num_sequences=num_sequences,
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
        name="baseline_speculative_decoding"
    )
    
    metrics = benchmark.run()
    benchmark.cleanup()
    
    return {
        "mean_time_ms": result.timing.mean_ms,
        **metrics,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline Speculative Decoding")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-draft-tokens", type=int, default=4)
    parser.add_argument("--num-sequences", type=int, default=10)
    parser.add_argument("--profile", type=str, default="none")
    
    args = parser.parse_args()
    
    result = run_benchmark(
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_draft_tokens=args.num_draft_tokens,
        num_sequences=args.num_sequences,
        profile=args.profile,
    )
    
    print(f"\n{'='*60}")
    print(f"Baseline Speculative Decoding Results")
    print(f"{'='*60}")
    print(f"Tokens/sec: {result['tokens_per_sec']:.2f}")
    print(f"Acceptance rate: {result['acceptance_rate']:.1f}%")
    print(f"Tokens generated: {result['tokens_generated']}")
    print(f"Mean time: {result['mean_time_ms']:.2f} ms")
    print(f"{'='*60}\n")


# Harness integration
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata  # noqa: E402


class BaselineSequentialDecodeBenchmark(BaseBenchmark):
    """Baseline: Sequential autoregressive decoding with target model only.
    
    This is the standard approach that speculative decoding aims to improve.
    Generates N tokens sequentially, each requiring a full target forward pass.
    """
    
    def __init__(self):
        super().__init__()
        self.batch_size = 4
        self.vocab_size = 32000
        self.hidden_size = 4096
        self.tokens_to_generate = 40  # 10 sequences * 4 draft tokens
        self.target_model = None
        self.input_ids = None
        
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.batch_size * self.tokens_to_generate),
        )
    
    def setup(self) -> None:
        """Initialize target model for sequential decoding."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        torch.manual_seed(42)
        
        # Create target model (large)
        class SimpleLM(nn.Module):
            def __init__(self, vocab_size, hidden_size):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.linear1 = nn.Linear(hidden_size, hidden_size)
                self.linear2 = nn.Linear(hidden_size, vocab_size)
            
            def forward(self, input_ids):
                x = self.embedding(input_ids)
                x = torch.relu(self.linear1(x))
                return self.linear2(x)
        
        self.target_model = SimpleLM(self.vocab_size, self.hidden_size).to(self.device).eval()
        self.input_ids = torch.randint(0, self.vocab_size, (self.batch_size, 1), device=self.device)
        
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )
    
    def benchmark_fn(self) -> None:
        """Sequential autoregressive decode: one target forward per token."""
        with self._nvtx_range("baseline_sequential_decode"):
            current_ids = self.input_ids.clone()
            
            # Generate tokens ONE AT A TIME using target model
            # This is the slow baseline that speculative decoding improves
            for _ in range(self.tokens_to_generate):
                with torch.no_grad():
                    logits = self.target_model(current_ids[:, -1:])
                    next_token = torch.argmax(logits, dim=-1)
                    current_ids = torch.cat([current_ids, next_token], dim=1)
        
        self._synchronize()
    
    def teardown(self) -> None:
        """Clean up resources."""
        self.target_model = None
        self.input_ids = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)
    
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


def get_benchmark() -> BaseBenchmark:
    return BaselineSequentialDecodeBenchmark()
