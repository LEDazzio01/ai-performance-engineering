#!/usr/bin/env python3
"""FlexAttention with document-level attention masks.

Demonstrates document-aware attention where tokens attend only within
their document boundaries, useful for multi-document batching.
"""

import torch
from typing import Dict, Any, Callable
import sys
from pathlib import Path

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.python.benchmark_harness import BenchmarkHarness, BenchmarkConfig, BenchmarkMode
from common.python.logger import get_logger

logger = get_logger(__name__)

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False


class DocumentAttentionFlexAttention:
    """Document-level attention with FlexAttention."""
    
    def __init__(
        self,
        batch_size: int = 4,
        num_heads: int = 32,
        seq_length: int = 8192,
        head_dim: int = 128,
        num_documents: int = 8,
        use_compile: bool = True,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.head_dim = head_dim
        self.num_documents = num_documents
        self.use_compile = use_compile
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.doc_length = seq_length // num_documents
        
        logger.info(f"Document Attention")
        logger.info(f"  Documents: {num_documents}, length: {self.doc_length}")
    
    def _create_document_mask(self) -> Callable:
        """Create document boundary mask."""
        doc_length = self.doc_length
        
        def document_pattern(b, h, q_idx, kv_idx):
            # Same document check
            q_doc = q_idx // doc_length
            kv_doc = kv_idx // doc_length
            return q_doc == kv_doc
        
        return document_pattern
    
    def setup(self):
        """Initialize FlexAttention."""
        self.q = torch.randn(
            self.batch_size, self.num_heads, self.seq_length, self.head_dim,
            device=self.device, dtype=torch.bfloat16
        )
        self.k = torch.randn_like(self.q)
        self.v = torch.randn_like(self.q)
        
        if FLEX_ATTENTION_AVAILABLE:
            mask_fn = self._create_document_mask()
            self.block_mask = create_block_mask(
                mask_fn,
                B=self.batch_size,
                H=self.num_heads,
                Q_LEN=self.seq_length,
                KV_LEN=self.seq_length,
                device=self.device
            )
            
            if self.use_compile:
                self.attention_fn = torch.compile(flex_attention)
            else:
                self.attention_fn = flex_attention
    
    def run(self) -> float:
        """Execute document attention."""
        if not FLEX_ATTENTION_AVAILABLE:
            return 0.0
        
        import time
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        output = self.attention_fn(self.q, self.k, self.v, block_mask=self.block_mask)
        
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000
    
    def cleanup(self):
        """Clean up."""
        del self.q, self.k, self.v
        if hasattr(self, 'block_mask'):
            del self.block_mask
        torch.cuda.empty_cache()


def run_benchmark(
    batch_size: int = 4,
    num_heads: int = 32,
    seq_length: int = 8192,
    num_documents: int = 8,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run document attention benchmark."""
    
    benchmark = DocumentAttentionFlexAttention(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_length=seq_length,
        num_documents=num_documents,
    )
    benchmark.setup()
    
    config = BenchmarkConfig(iterations=5, warmup=10, profile_mode=profile)
    harness = BenchmarkHarness(mode=BenchmarkMode.INFERENCE, config=config)
    
    result = harness.benchmark(benchmark.run, name="flexattention_document")
    benchmark.cleanup()
    
    return {
        "mean_time_ms": result.timing.mean_ms,
        "num_documents": num_documents,
        "doc_length": seq_length // num_documents,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FlexAttention Document")
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--num-documents", type=int, default=8)
    parser.add_argument("--profile", type=str, default="none")
    
    args = parser.parse_args()
    
    result = run_benchmark(
        seq_length=args.seq_length,
        num_documents=args.num_documents,
        profile=args.profile,
    )
    
    print(f"\n{'='*60}")
    print(f"Document Attention Results")
    print(f"{'='*60}")
    print(f"Documents: {result['num_documents']}")
    print(f"Doc length: {result['doc_length']}")
    print(f"Mean time: {result['mean_time_ms']:.2f} ms")
    print(f"{'='*60}\n")
    print(f"Use case: Multi-document batching, RAG pipelines")





