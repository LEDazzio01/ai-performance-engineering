#!/usr/bin/env python3
"""FlexAttention with block-sparse attention patterns.

Demonstrates block-sparse attention using FlexAttention for efficient
long-context processing on Blackwell.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable
import sys
from pathlib import Path
import math

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.python.benchmark_harness import BenchmarkHarness, BenchmarkConfig, BenchmarkMode
from common.python.logger import get_logger

logger = get_logger(__name__)

# Check for FlexAttention
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    logger.warning("FlexAttention not available, using fallback")


class BlockSparseFlexAttention:
    """Block-sparse attention with FlexAttention."""
    
    def __init__(
        self,
        batch_size: int = 2,
        num_heads: int = 32,
        seq_length: int = 8192,
        head_dim: int = 128,
        block_size: int = 256,
        use_compile: bool = True,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.head_dim = head_dim
        self.block_size = block_size
        self.use_compile = use_compile
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Block-sparse FlexAttention")
        logger.info(f"  Sequence: {seq_length}, Block size: {block_size}")
        logger.info(f"  Sparsity: ~{self._calculate_sparsity():.1f}%")
    
    def _calculate_sparsity(self) -> float:
        """Calculate expected sparsity percentage."""
        # Block-sparse pattern: attend to local block + global tokens
        num_blocks = self.seq_length // self.block_size
        local_attention = self.block_size * self.block_size  # Within block
        global_attention = self.block_size * num_blocks  # To global tokens
        
        total_attention = local_attention + global_attention
        full_attention = self.seq_length * self.seq_length
        
        sparsity = (1 - total_attention / full_attention) * 100
        return sparsity
    
    def _create_block_sparse_mask(self) -> Callable:
        """Create block-sparse mask function.
        
        Pattern: Each block attends to:
        1. All tokens within its own block
        2. First token of every other block (global tokens)
        """
        block_size = self.block_size
        
        def block_sparse_pattern(b, h, q_idx, kv_idx):
            # Same block (local attention)
            same_block = (q_idx // block_size) == (kv_idx // block_size)
            
            # Global tokens (first token of each block)
            is_global = (kv_idx % block_size) == 0
            
            return same_block | is_global
        
        return block_sparse_pattern
    
    def setup(self):
        """Initialize FlexAttention with block-sparse pattern."""
        # Create Q, K, V
        self.q = torch.randn(
            self.batch_size,
            self.num_heads,
            self.seq_length,
            self.head_dim,
            device=self.device,
            dtype=torch.bfloat16
        )
        self.k = torch.randn_like(self.q)
        self.v = torch.randn_like(self.q)
        
        if FLEX_ATTENTION_AVAILABLE:
            # Create block-sparse mask
            mask_fn = self._create_block_sparse_mask()
            self.block_mask = create_block_mask(
                mask_fn,
                B=self.batch_size,
                H=self.num_heads,
                Q_LEN=self.seq_length,
                KV_LEN=self.seq_length,
                device=self.device
            )
            
            # Compile if requested
            if self.use_compile:
                logger.info("Compiling FlexAttention...")
                self.attention_fn = torch.compile(flex_attention)
            else:
                self.attention_fn = flex_attention
            
            logger.info("FlexAttention setup complete")
        else:
            logger.info("Using fallback attention (FlexAttention not available)")
    
    def run(self) -> float:
        """Execute block-sparse attention."""
        if not FLEX_ATTENTION_AVAILABLE:
            # Fallback: standard attention
            import time
            start = time.perf_counter()
            
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(self.q, self.k.transpose(-2, -1)) * scale
            attn = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn, self.v)
            
            torch.cuda.synchronize()
            return (time.perf_counter() - start) * 1000
        
        torch.cuda.synchronize()
        import time
        start = time.perf_counter()
        
        # FlexAttention with block-sparse mask
        output = self.attention_fn(
            self.q,
            self.k,
            self.v,
            block_mask=self.block_mask
        )
        
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        logger.info(f"Block-sparse attention: {elapsed_ms:.2f} ms")
        
        return elapsed_ms
    
    def cleanup(self):
        """Clean up resources."""
        del self.q, self.k, self.v
        if hasattr(self, 'block_mask'):
            del self.block_mask
        torch.cuda.empty_cache()


def run_benchmark(
    batch_size: int = 2,
    num_heads: int = 32,
    seq_length: int = 8192,
    head_dim: int = 128,
    block_size: int = 256,
    use_compile: bool = True,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run block-sparse FlexAttention benchmark."""
    
    benchmark = BlockSparseFlexAttention(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_length=seq_length,
        head_dim=head_dim,
        block_size=block_size,
        use_compile=use_compile,
    )
    benchmark.setup()
    
    config = BenchmarkConfig(
        iterations=5,
        warmup=10,
        profile_mode=profile,
    )
    
    harness = BenchmarkHarness(mode=BenchmarkMode.INFERENCE, config=config)
    
    result = harness.benchmark(
        benchmark.run,
        name="flexattention_block_sparse"
    )
    
    benchmark.cleanup()
    
    return {
        "mean_time_ms": result.timing.mean_ms,
        "seq_length": seq_length,
        "block_size": block_size,
        "sparsity_pct": benchmark._calculate_sparsity(),
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FlexAttention Block-Sparse")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--profile", type=str, default="none")
    
    args = parser.parse_args()
    
    result = run_benchmark(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_length=args.seq_length,
        head_dim=args.head_dim,
        block_size=args.block_size,
        use_compile=not args.no_compile,
        profile=args.profile,
    )
    
    print(f"\n{'='*60}")
    print(f"FlexAttention Block-Sparse Results")
    print(f"{'='*60}")
    print(f"Sequence length: {result['seq_length']:,}")
    print(f"Block size: {result['block_size']}")
    print(f"Sparsity: {result['sparsity_pct']:.1f}%")
    print(f"Mean time: {result['mean_time_ms']:.2f} ms")
    print(f"{'='*60}\n")
    print(f"Expected: ~3-5× faster than dense attention for long sequences")





