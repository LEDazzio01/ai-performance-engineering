#!/usr/bin/env python3
"""Base benchmark class for MoE Optimization Journey.

Each level inherits from this and sets its LEVEL constant.
Optimizations are applied cumulatively based on level.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from labs.moe_optimization_journey.moe_model import (
    ConfigurableMoEModel,
    MoEOptimizations,
    create_model,
)


# Optimization descriptions for each level
LEVEL_DESCRIPTIONS = {
    0: ("Naive", "Python loops over experts"),
    1: ("+ Batched", "Einsum parallelizes all tokens"),
    2: ("+ Fused", "Triton kernel fuses SiLU*up"),
    3: ("+ MemEfficient", "Reuse buffers, reduce allocations"),
    4: ("+ Grouped", "Sort tokens + per-expert GEMM"),
    5: ("+ BMM Fusion", "Vectorized scatter + single BMM (5-6x!)"),  # NEW!
    6: ("+ CUDAGraphs", "Capture kernel sequence"),
    7: ("+ Compiled", "torch.compile does ALL of the above!"),
}


class MoEJourneyBenchmark(BaseBenchmark):
    """Base benchmark for MoE optimization journey.
    
    Subclasses just need to set LEVEL class variable.
    """
    
    LEVEL: int = 0  # Override in subclasses
    
    # Model configuration - Llama-7B like dimensions for realistic GPU utilization!
    VOCAB_SIZE = 32000
    HIDDEN_SIZE = 4096       # Llama-7B: 4096
    INTERMEDIATE_SIZE = 11008  # Llama-7B: 11008
    NUM_LAYERS = 1           # Just 1 layer for benchmarking MoE
    NUM_HEADS = 32
    NUM_EXPERTS = 8
    NUM_EXPERTS_PER_TOK = 2
    BATCH_SIZE = 16   # 16x512 = 8K tokens
    SEQ_LEN = 512
    
    WARMUP = 3
    ITERATIONS = 10
    
    def __init__(self):
        super().__init__()
        self.model: Optional[Any] = None
        self.compiled_model: Optional[Any] = None
        self.input_ids: Optional[torch.Tensor] = None
        self.opts: Optional[MoEOptimizations] = None
        self.last_latency_ms: float = 0.0
        self.last_tokens_per_sec: float = 0.0
        
        total_tokens = self.BATCH_SIZE * self.SEQ_LEN
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.BATCH_SIZE),
            tokens_per_iteration=float(total_tokens),
        )
    
    def setup(self) -> None:
        import gc
        
        # Clean up CUDA graph state from previous benchmarks
        # to prevent "Offset increment outside graph capture" errors
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            try:
                if hasattr(torch.cuda, 'graph_pool_trim'):
                    torch.cuda.graph_pool_trim()
            except Exception:
                pass
            
            # Reset CUDA RNG state to prevent graph capture errors
            try:
                device_idx = torch.cuda.current_device()
                gen = torch.cuda.default_generators[device_idx]
                gen.set_offset(0)
                gen.manual_seed(0)
            except Exception:
                pass
            
            try:
                torch._dynamo.reset()
            except Exception:
                pass
            
            try:
                torch._inductor.cudagraph_trees.reset_cudagraph_trees()
            except Exception:
                pass
        
        level = self.LEVEL
        name, desc = LEVEL_DESCRIPTIONS.get(level, (f"Level {level}", ""))
        
        print("=" * 60)
        print(f"LEVEL {level}: {name}")
        print("=" * 60)
        print(f"  {desc}")
        print()
        
        # Show cumulative optimizations
        print("  Optimizations enabled:")
        for l in range(level + 1):
            _, opt_desc = LEVEL_DESCRIPTIONS.get(l, ("", ""))
            if l == 0:
                print(f"    Level 0: {opt_desc}")
            else:
                print(f"    Level {l}: {opt_desc}")
        print()
        
        # Create model with optimizations up to this level
        self.model, self.opts = create_model(
            level=level,
            vocab_size=self.VOCAB_SIZE,
            hidden_size=self.HIDDEN_SIZE,
            intermediate_size=self.INTERMEDIATE_SIZE,
            num_layers=self.NUM_LAYERS,
            num_heads=self.NUM_HEADS,
            num_experts=self.NUM_EXPERTS,
            num_experts_per_tok=self.NUM_EXPERTS_PER_TOK,
        )
        self.model = self.model.to(self.device).to(torch.bfloat16)
        self.model.eval()
        
        params = sum(p.numel() for p in self.model.parameters())
        print(f"  Parameters: {params / 1e6:.1f}M")
        print(f"  Batch: {self.BATCH_SIZE} x {self.SEQ_LEN} = {self.BATCH_SIZE * self.SEQ_LEN} tokens")
        
        # Apply torch.compile if enabled (Level 5)
        if self.opts.use_compile:
            # Always use max-autotune for best performance
            print(f"\n  Compiling with mode='max-autotune'...")
            self.compiled_model = torch.compile(self.model, mode="max-autotune")
        else:
            self.compiled_model = self.model
        
        # Create input using CPU random + to(device) to avoid CUDA RNG graph capture issues
        self.input_ids = torch.randint(
            0, self.VOCAB_SIZE,
            (self.BATCH_SIZE, self.SEQ_LEN),
        ).to(self.device)
        
        # Warmup
        print(f"\n  Warmup ({self.WARMUP + 2} iterations)...")
        for i in range(self.WARMUP + 2):
            with torch.no_grad():
                _ = self.compiled_model(self.input_ids)
            if i == 0 and self.opts.use_compile:
                print("    First run (compile): done")
        torch.cuda.synchronize()
        print("  Ready")
    
    def benchmark_fn(self) -> None:
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with self._nvtx_range(f"level{self.LEVEL}"):
            with torch.no_grad():
                _ = self.compiled_model(self.input_ids)
        
        torch.cuda.synchronize()
        self.last_latency_ms = (time.perf_counter() - start) * 1000
        
        total_tokens = self.BATCH_SIZE * self.SEQ_LEN
        self.last_tokens_per_sec = total_tokens / (self.last_latency_ms / 1000)
    
    def teardown(self) -> None:
        del self.compiled_model
        del self.model
        self.compiled_model = None
        self.model = None
        self.input_ids = None
        torch.cuda.empty_cache()
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=self.ITERATIONS,
            warmup=self.WARMUP,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def validate_result(self) -> Optional[str]:
        return None if self.compiled_model else "Model not initialized"
    
    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return {
            "level": float(self.LEVEL),
            "latency_ms": self.last_latency_ms,
            "tokens_per_sec": self.last_tokens_per_sec,
            "use_batched": float(self.opts.use_batched if self.opts else 0),
            "use_fused": float(self.opts.use_fused if self.opts else 0),
            "use_mem_efficient": float(self.opts.use_mem_efficient if self.opts else 0),
            "use_grouped": float(self.opts.use_grouped if self.opts else 0),
            "use_cuda_graphs": float(self.opts.use_cuda_graphs if self.opts else 0),
            "use_compile": float(self.opts.use_compile if self.opts else 0),
        }
    
    def get_input_signature(self) -> Optional[Dict[str, Any]]:
        """Return input signature for verification.
        
        Captures MoE-specific workload parameters to ensure baseline and
        optimized benchmarks operate on equivalent workloads.
        """
        return {
            "batch_size": self.BATCH_SIZE,
            "seq_len": self.SEQ_LEN,
            "hidden_size": self.HIDDEN_SIZE,
            "intermediate_size": self.INTERMEDIATE_SIZE,
            "num_experts": self.NUM_EXPERTS,
            "num_experts_per_tok": self.NUM_EXPERTS_PER_TOK,
            "vocab_size": self.VOCAB_SIZE,
            "num_heads": self.NUM_HEADS,
        }


def run_level(level: int) -> None:
    """Run a specific level benchmark."""
    class LevelBenchmark(MoEJourneyBenchmark):
        LEVEL = level
    
    benchmark = LevelBenchmark()
    benchmark.setup()
    
    times = []
    for i in range(5):
        benchmark.benchmark_fn()
        times.append(benchmark.last_latency_ms)
        print(f"  Run {i+1}: {benchmark.last_latency_ms:.1f} ms ({benchmark.last_tokens_per_sec:,.0f} tok/s)")
    
    avg = sum(times) / len(times)
    print(f"\nMean: {avg:.1f} ms")
    benchmark.teardown()
    return avg


if __name__ == "__main__":
    import sys
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_level(level)

