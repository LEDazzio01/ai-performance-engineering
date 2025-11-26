# 🚀 Auto-Optimizer

**LLM-powered GPU code optimization tool** that automatically profiles, analyzes, and optimizes PyTorch/CUDA code.

## Features

- **Multi-source input**: Optimize local files, GitHub repos, or existing benchmark pairs
- **Intelligent profiling**: Automatic bottleneck detection using torch.profiler
- **LLM-powered analysis**: Uses Claude/GPT to suggest optimizations
- **Iterative refinement**: Multiple optimization passes with validation
- **Safe patching**: AST-based code modification with syntax validation

## Quick Start

```bash
# Optimize a single file
python -m tools.optimize model.py --output optimized_model.py

# Optimize from a GitHub repo
python -m tools.optimize https://github.com/user/repo --target src/model.py

# Scan and optimize all underperforming benchmarks
python -m tools.optimize --scan . --threshold 1.1

# Use OpenAI instead of Anthropic
python -m tools.optimize model.py --provider openai --model gpt-4o
```

## CLI Reference

```
usage: python -m tools.optimize [-h] [--output OUTPUT] [--target TARGET]
                                [--scan] [--threshold THRESHOLD]
                                [--provider {anthropic,openai}] [--model MODEL]
                                [--iterations ITERATIONS] [--verbose] [--quiet]
                                [input]

positional arguments:
  input                 File path, directory, or repo URL to optimize

options:
  -h, --help            Show help message
  --output, -o OUTPUT   Output file/directory path
  --target, -t TARGET   Target files in repo (can specify multiple)
  --scan                Scan directory for benchmark pairs to optimize
  --threshold THRESHOLD Speedup threshold for --scan (default: 1.1)
  --provider            LLM provider: anthropic or openai (default: anthropic)
  --model MODEL         LLM model to use
  --iterations N        Max optimization iterations (default: 3)
  --verbose, -v         Verbose output (default: true)
  --quiet, -q           Suppress output
```

## Python API

### Basic Usage

```python
from tools.optimize import AutoOptimizer

optimizer = AutoOptimizer(
    llm_provider="anthropic",
    model="claude-sonnet-4-20250514",
    max_iterations=3,
    target_speedup=1.2,
)

# Optimize a single file
result = optimizer.optimize_file(
    "model.py",
    output_path="optimized_model.py"
)

print(f"Speedup: {result.speedup:.2f}x")
print(f"Techniques: {result.techniques_applied}")
print(result.explanation)
```

### Optimize a GitHub Repository

```python
results = optimizer.optimize_repo(
    "https://github.com/user/ml-project",
    target_files=["src/model.py", "src/train.py"],
    branch="main",
    output_dir="./optimized/"
)

for file_path, result in results.items():
    print(f"{file_path}: {result.speedup:.2f}x")
```

### Scan and Optimize Benchmarks

```python
results = optimizer.scan_and_optimize(
    directory=".",
    threshold=1.05,  # Only optimize if current speedup < 1.05x
    pattern="optimized_*.py"
)
```

## Input Adapters

The optimizer supports multiple input sources through adapters:

### FileAdapter
```python
from tools.optimize import FileAdapter

adapter = FileAdapter(
    paths=["model1.py", "model2.py"],
    output_dir="./optimized/",
    suffix="_optimized"
)
```

### RepoAdapter
```python
from tools.optimize import RepoAdapter

adapter = RepoAdapter(
    repo_url="https://github.com/user/repo",
    target_files=["src/model.py"],  # Optional, auto-detects GPU files
    branch="main"
)
```

### BenchmarkAdapter
```python
from tools.optimize import BenchmarkAdapter

adapter = BenchmarkAdapter(
    directory="./benchmarks/",
    threshold=1.1,
    pattern="optimized_*.py"
)
```

## Configuration File

Create `optimize_config.yaml` for persistent settings:

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  max_tokens: 16384

optimization:
  max_iterations: 3
  target_speedup: 1.2
  techniques:
    - torch_compile
    - mixed_precision
    - cuda_graphs
    - kernel_fusion

profiling:
  warmup_iterations: 3
  benchmark_iterations: 10
  
output:
  save_intermediate: true
  generate_report: true
```

## Output

### OptimizationResult

```python
@dataclass
class OptimizationResult:
    success: bool              # Whether optimization improved performance
    original_code: str         # Original source code
    optimized_code: str        # Optimized source code
    original_time_ms: float    # Original execution time
    optimized_time_ms: float   # Optimized execution time
    speedup: float             # Performance improvement ratio
    techniques_applied: List[str]  # Applied optimization techniques
    explanation: str           # Human-readable explanation
    profile_data: Dict         # Detailed profiling data
    patches_applied: List[Dict]    # Applied code patches
    errors: List[str]          # Any errors encountered
```

## Optimization Techniques

The optimizer can apply various GPU optimization techniques:

| Technique | Description |
|-----------|-------------|
| `torch.compile` | JIT compilation with kernel fusion |
| Mixed Precision | BF16/FP16 for Tensor Core utilization |
| CUDA Graphs | Reduce kernel launch overhead |
| Memory Optimization | Efficient memory access patterns |
| Kernel Fusion | Combine multiple operations |
| Async Operations | Overlap compute and memory transfers |

## Environment Variables

```bash
# LLM API keys
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# Cache directory
export AUTO_OPTIMIZER_CACHE="~/.cache/auto-optimizer"

# Logging level
export AUTO_OPTIMIZER_LOG_LEVEL="INFO"
```

## Examples

See [`examples/optimize_examples.py`](../../examples/optimize_examples.py) for more usage examples.

## Troubleshooting

### Common Issues

1. **"No benchmark class found"**: Ensure your file has a class inheriting from `BaseBenchmark`
2. **"LLM analysis failed"**: Check API key and network connection
3. **"Patch application failed"**: Review the generated patch for syntax issues

### Debug Mode

```bash
python -m tools.optimize model.py --verbose 2>&1 | tee optimize.log
```

## Architecture

```
tools/optimize/
├── __init__.py          # Public API exports
├── __main__.py          # CLI entry point
├── optimizer.py         # Core AutoOptimizer class
├── input_adapters.py    # File/Repo/Benchmark adapters
└── README.md            # This file
```

