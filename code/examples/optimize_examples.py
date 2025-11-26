#!/usr/bin/env python3
"""
Auto-Optimizer Usage Examples

Demonstrates how to use the auto-optimizer to improve GPU code performance.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn


# =============================================================================
# Example 1: Optimize a Simple Model
# =============================================================================

def example_optimize_model():
    """Optimize a PyTorch model file."""
    from tools.optimize import AutoOptimizer
    
    print("=" * 60)
    print("Example 1: Optimize a Model File")
    print("=" * 60)
    
    # Create optimizer
    optimizer = AutoOptimizer(
        llm_provider="anthropic",
        max_iterations=2,
        target_speedup=1.2,
        verbose=True,
    )
    
    # Create a sample model file to optimize
    sample_code = '''
import torch
import torch.nn as nn
from tools.testing.benchmark_harness import BaseBenchmark

class SimpleModel(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # Inefficient: separate operations
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.ln(x)
        return x

class ModelBenchmark(BaseBenchmark):
    def setup(self):
        self.model = SimpleModel().cuda()
        self.input = torch.randn(32, 1024, device='cuda')
        
    def benchmark_fn(self):
        return self.model(self.input)

def get_benchmark():
    return ModelBenchmark()
'''
    
    # Write sample file
    sample_file = Path("/tmp/sample_model.py")
    sample_file.write_text(sample_code)
    
    # Optimize
    result = optimizer.optimize_file(
        sample_file,
        output_path=Path("/tmp/optimized_model.py"),
    )
    
    print(f"\nResult:")
    print(f"  Success: {result.success}")
    print(f"  Speedup: {result.speedup:.2f}x")
    print(f"  Original time: {result.original_time_ms:.2f}ms")
    print(f"  Optimized time: {result.optimized_time_ms:.2f}ms")
    print(f"  Techniques: {result.techniques_applied}")
    print(f"\nExplanation:\n{result.explanation}")


# =============================================================================
# Example 2: Scan and Optimize Benchmarks
# =============================================================================

def example_scan_benchmarks():
    """Scan a directory and optimize underperforming benchmarks."""
    from tools.optimize import AutoOptimizer
    
    print("\n" + "=" * 60)
    print("Example 2: Scan and Optimize Benchmarks")
    print("=" * 60)
    
    optimizer = AutoOptimizer(
        llm_provider="anthropic",
        max_iterations=2,
        verbose=True,
    )
    
    # Scan current directory for benchmarks with speedup < 1.1x
    # In a real scenario, this would scan your benchmark directory
    print("\nScanning for underperforming benchmarks...")
    print("(This would scan ch*/ and labs/ directories)")
    
    # Example of what the scan would return
    print("""
Example output:
  ch5:matmul_tiling: current speedup 1.02x < 1.1x
    ✅ Optimized: 1.02x → 1.45x
    Techniques: torch.compile, memory_coalescing
    
  ch7:async_prefetch: current speedup 1.08x < 1.1x
    ✅ Optimized: 1.08x → 1.32x
    Techniques: cuda_streams, prefetching
""")


# =============================================================================
# Example 3: Optimize from GitHub Repository
# =============================================================================

def example_optimize_repo():
    """Optimize code from a GitHub repository."""
    from tools.optimize import AutoOptimizer
    
    print("\n" + "=" * 60)
    print("Example 3: Optimize from GitHub Repository")
    print("=" * 60)
    
    print("""
Usage:
    optimizer = AutoOptimizer()
    
    results = optimizer.optimize_repo(
        "https://github.com/user/ml-project",
        target_files=["src/model.py", "src/train.py"],
        branch="main",
        output_dir="./optimized/"
    )
    
    for file_path, result in results.items():
        print(f"{file_path}: {result.speedup:.2f}x")
        
Example output:
    Cloning https://github.com/user/ml-project...
    Found 3 GPU-related files
    
    Optimizing src/model.py...
      ✅ 1.82x speedup
      Techniques: torch.compile, mixed_precision
      
    Optimizing src/train.py...
      ✅ 1.45x speedup
      Techniques: cuda_graphs, gradient_checkpointing
""")


# =============================================================================
# Example 4: Using Input Adapters
# =============================================================================

def example_input_adapters():
    """Demonstrate different input adapter types."""
    from tools.optimize.input_adapters import FileAdapter, BenchmarkAdapter, detect_input_type
    
    print("\n" + "=" * 60)
    print("Example 4: Input Adapters")
    print("=" * 60)
    
    # Auto-detect input type
    print("\nAuto-detection examples:")
    
    test_inputs = [
        "model.py",
        "https://github.com/user/repo",
        "./benchmarks/",
        "-",  # stdin
    ]
    
    for inp in test_inputs:
        try:
            input_type, adapter = detect_input_type(inp)
            print(f"  '{inp}' → {input_type} ({adapter.__class__.__name__})")
        except Exception as e:
            print(f"  '{inp}' → Error: {e}")
    
    # File adapter
    print("\nFileAdapter:")
    print("""
    adapter = FileAdapter(
        paths=["model1.py", "model2.py"],
        output_dir="./optimized/",
        suffix="_optimized"
    )
    
    for source in adapter.get_sources():
        print(f"Processing: {source.name}")
        # ... optimize ...
        adapter.write_output(source, optimized_code)
""")
    
    # Benchmark adapter
    print("\nBenchmarkAdapter:")
    print("""
    adapter = BenchmarkAdapter(
        directory="./ch5/",
        threshold=1.1,
        pattern="optimized_*.py"
    )
    
    for source in adapter.get_sources():
        print(f"Benchmark: {source.name}")
        # Only yields files where current speedup < threshold
""")


# =============================================================================
# Example 5: Configuration File
# =============================================================================

def example_config_file():
    """Show configuration file usage."""
    
    print("\n" + "=" * 60)
    print("Example 5: Configuration File")
    print("=" * 60)
    
    config_example = """
# optimize_config.yaml

llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  max_tokens: 16384
  temperature: 0.1

optimization:
  max_iterations: 3
  target_speedup: 1.2
  techniques:
    - torch_compile
    - mixed_precision
    - cuda_graphs
    - kernel_fusion
    - memory_optimization

profiling:
  warmup_iterations: 3
  benchmark_iterations: 10
  enable_memory_tracking: true
  
output:
  save_intermediate: true
  generate_report: true
  output_format: markdown

# Technique-specific settings
torch_compile:
  mode: reduce-overhead
  fullgraph: false
  
mixed_precision:
  dtype: bfloat16
  enabled_ops:
    - linear
    - matmul
    - conv2d
"""
    
    print(config_example)
    
    print("""
Usage:
    optimizer = AutoOptimizer.from_config("optimize_config.yaml")
    result = optimizer.optimize_file("model.py")
""")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("🚀 Auto-Optimizer Examples\n")
    
    # Run examples that don't require actual optimization
    example_input_adapters()
    example_config_file()
    example_scan_benchmarks()
    example_optimize_repo()
    
    # Only run actual optimization if explicitly requested
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-optimization", action="store_true",
                       help="Run the actual optimization example (requires LLM API key)")
    args = parser.parse_args()
    
    if args.run_optimization:
        example_optimize_model()
    else:
        print("\n" + "=" * 60)
        print("To run actual optimization examples, use: --run-optimization")
        print("(Requires ANTHROPIC_API_KEY or OPENAI_API_KEY)")
        print("=" * 60)

