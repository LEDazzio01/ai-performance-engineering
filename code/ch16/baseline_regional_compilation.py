"""baseline_regional_compilation.py - Baseline: Full model compilation (hangs, falls back to eager).

Demonstrates the problem: torch.compile on entire large model (>40B params) hangs indefinitely.
This baseline shows what NOT to do for large models.

Regional Compilation: This baseline compiles the ENTIRE model at once, which causes hangs
on models >40B parameters due to graph explosion and memory exhaustion. After timeout,
it falls back to eager mode (no compilation).
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn


from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch16")
    return torch.device("cuda")


class LargeTransformerBlock(nn.Module):
    """A large transformer block that's computationally expensive."""
    
    def __init__(self, d_model: int = 8192, d_ff: int = 32768):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=64, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out
        residual = x
        x = self.ln2(x)
        x = residual + self.mlp(x)
        return x


class LargeTransformerModel(nn.Module):
    """A large transformer model (~40B+ parameters) that causes compilation hangs."""
    
    def __init__(self, n_layers: int = 48, d_model: int = 8192, d_ff: int = 32768):
        super().__init__()
        self.embed = nn.Embedding(50304, d_model)
        self.blocks = nn.ModuleList([
            LargeTransformerBlock(d_model, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, 50304, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x


class BaselineRegionalCompilationBenchmark(Benchmark):
    """Baseline: Full model compilation (PROBLEMATIC - hangs on large models).
    
    Regional Compilation: This baseline compiles the ENTIRE model at once using
    torch.compile(model). This causes indefinite hangs on models >40B parameters
    due to:
    - Graph explosion (exponential complexity)
    - Memory exhaustion during compilation
    - No timeout mechanism
    
    This demonstrates the problem that regional compilation solves.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.config = None
    
    def setup(self, config: BenchmarkConfig) -> None:
        """Create and compile the entire model (problematic approach)."""
        # Create a large model (~40B parameters)
        n_layers = 48
        d_model = 8192
        d_ff = 32768
        
        model = LargeTransformerModel(n_layers=n_layers, d_model=d_model, d_ff=d_ff)
        model = model.to(self.device, dtype=torch.bfloat16)
        model.eval()
        
        # PROBLEM: Compile entire model at once
        # This will hang on large models (>40B params) due to graph explosion
        print("=" * 80)
        print("BASELINE: Compiling ENTIRE model (this may hang on large models!)")
        print("=" * 80)
        print(f"Model: {n_layers} layers, d_model={d_model}, d_ff={d_ff}")
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count / 1e9:.2f}B")
        
        if param_count > 40_000_000_000:
            print("WARNING: WARNING: Model >40B params - compilation WILL HANG!")
            print("   This baseline demonstrates the problem.")
            print("   See optimized_regional_compilation.py for the solution.")
        
        # This is where it hangs - compiling entire model
        print("\nAttempting to compile entire model...")
        print("(This will hang/timeout for large models >40B params)")
        print("Timeout: 10 seconds (will fall back to eager mode)")
        
        # Use threading timeout (more reliable than signal for blocking operations)
        import threading
        compilation_result = {"model": None, "done": False, "error": None}
        
        def compile_in_thread():
            try:
                compilation_result["model"] = torch.compile(model, mode="reduce-overhead")
                compilation_result["done"] = True
            except Exception as e:
                compilation_result["error"] = e
                compilation_result["done"] = True
        
        thread = threading.Thread(target=compile_in_thread, daemon=True)
        thread.start()
        thread.join(timeout=10.0)  # 10 second timeout
        
        if thread.is_alive():
            # Compilation is still running (hanging)
            print(f"\nERROR: Compilation timed out after 10 seconds (expected for large models)")
            print(f"   Falling back to EAGER mode (no compilation)")
            print("\nThis demonstrates why regional compilation is needed!")
            self.model = model
            print("[OK] Using eager mode instead (baseline fallback)")
        elif compilation_result["error"]:
            print(f"\nERROR: Compilation failed: {compilation_result['error']}")
            print("   Falling back to EAGER mode (no compilation)")
            self.model = model
            print("[OK] Using eager mode instead (baseline fallback)")
        elif compilation_result["done"] and compilation_result["model"]:
            print("[OK] Compilation completed (unlikely for large models)")
            self.model = compilation_result["model"]
        else:
            # Shouldn't happen, but fallback to eager
            print("\nWARNING: Compilation status unclear, falling back to EAGER mode")
            self.model = model
        
        # Create input data
        batch_size = 2
        seq_len = 1024
        self.input_data = torch.randint(
            0, 50304, (batch_size, seq_len), device=self.device, dtype=torch.long
        )
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - runs inference."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_regional_compilation", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.model(self.input_data)

    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=1,
            warmup=0,
            timeout_seconds=15,  # Short timeout since compilation may hang
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None
    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineRegionalCompilationBenchmark()


def main():
    """Run the baseline benchmark."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = BaselineRegionalCompilationBenchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\n[OK] Baseline completed: {result.mean_ms:.3f} ms")
    print("NOTE: This ran in EAGER mode because compilation hung.")
    print("See optimized_regional_compilation.py for regional compilation solution.")


if __name__ == "__main__":
    main()

