# CUTLASS Backend Setup for PyTorch

## Overview

CUTLASS (CUDA Templates for Linear Algebra Subroutines) is NVIDIA's high-performance GEMM library integrated into PyTorch 2.9's TorchInductor. This guide shows how to enable and use it.

## Status

✅ **CUTLASS backend is fully working** (October 28, 2025)

- ✅ Installed system-wide via `nvidia-cutlass-dsl` package (`v4.2.1`)
- ✅ No import errors (path issues fixed)
- ✅ No TF32 API conflicts (legacy API removed)
- ✅ Available in torch.compile via `max-autotune` mode
- ⚠️ Performance limited by memory bandwidth (not a CUTLASS issue)

## Installation

### Quick Setup

Run the setup script (includes CUTLASS backend):

```bash
sudo ./setup.sh
```

This automatically installs `nvidia-cutlass-dsl` and `cuda-python` system-wide.

### Manual Installation

If you prefer to install manually:

```bash
sudo pip install nvidia-cutlass-dsl==4.2.1 cuda-python
```

**Why system-wide?** PyTorch's compilation workers need access to these packages. User-local installations (`~/.local`) may not be visible to worker processes.

**Note:** We use the pip package, not the git submodule. The old `third_party/cutlass` directory has been removed (saved 210MB).

### Verification

```python
import arch_config  # Auto-configures CUTLASS
import torch

cfg = torch._inductor.config
print(f"Backends: {cfg.max_autotune_gemm_backends}")
# Output: CUTLASS,TRITON,ATEN

print(f"CUTLASS ops: {cfg.cuda.cutlass_enabled_ops}")
# Output: all
```

## Configuration

The `arch_config.py` module automatically configures CUTLASS:

```python
# Enabled backends (CUTLASS first for priority)
cfg.max_autotune_gemm_backends = "CUTLASS,TRITON,ATEN"

# Enable CUTLASS for all operations
cfg.cuda.cutlass_enabled_ops = "all"

# Configure CUTLASS directory
cfg.cuda.cutlass_dir = "/path/to/cutlass"
```

## Usage

### Basic Example

```python
import arch_config  # Important: import first!
import torch
import torch.nn as nn

model = nn.Linear(2048, 4096).cuda().half()

# Compile with CUTLASS backend available
compiled_model = torch.compile(
    model,
    mode='max-autotune',  # Required for CUTLASS
    fullgraph=True,
)

x = torch.randn(64, 2048, device='cuda', dtype=torch.float16)
output = compiled_model(x)
```

### Compilation Modes

| Mode | CUTLASS | Description |
|------|---------|-------------|
| `default` | ❌ | Fast compilation, no auto-tuning |
| `reduce-overhead` | ❌ | Optimizes overhead, no GEMM tuning |
| `max-autotune` | ✅ | **Enables CUTLASS**, benchmarks backends |
| `max-autotune-no-cudagraphs` | ✅ | CUTLASS without CUDA graphs |

**Key:** Only `max-autotune` modes will attempt to use CUTLASS kernels.

## Performance Expectations

### When CUTLASS Helps

✅ **Compute-bound operations:**
- Large GEMM operations (M, N, K > 1024)
- FP32 matmuls (TF32 kernels)
- Batch sizes that saturate GPU

### When CUTLASS Doesn't Help

❌ **Memory-bound operations:**
- Small matrices (M, N, K < 512)
- FP16 on modern GPUs (already very fast)
- Large models (40B+) where memory bandwidth dominates
- Low batch sizes

### Example Benchmarks

#### Actual Benchmark Results (B200)

#### Small Model (16M params, FP16)
```
Eager:    0.065 ms
Compiled: 0.124 ms
Speedup:  0.52x  ❌ Slower (memory-bound + compilation overhead)
```

#### Medium Model (67M params, FP16)
```
Eager:    0.132 ms
Compiled: 0.199 ms
Speedup:  0.66x  ❌ Slower (memory-bound)
```

#### Large Model (268M params, FP32 with TF32)
```
Eager:    0.670 ms
Compiled: 0.728 ms
Speedup:  0.92x  ⚠️ Close to parity (still memory-bound)
```

#### Very Large Model (40B params, FP16)
```
Eager:    ~80 ms
Compiled: ~82 ms
Speedup:  0.98x  ⚠️ Memory bandwidth limit (cannot be overcome)
```

## Troubleshooting

### "Failed to import CUTLASS lib"

**Cause:** `nvidia-cutlass-dsl` or `cuda-python` not installed system-wide.

**Fix:**
```bash
# Remove user-installed versions
pip uninstall -y nvidia-cutlass-dsl cuda-python

# Install system-wide
sudo pip install nvidia-cutlass-dsl cuda-python
```

### "ModuleNotFoundError: No module named 'cuda.bindings'"

**Cause:** `cuda-python` package missing or in wrong location.

**Fix:**
```bash
sudo pip install --force-reinstall cuda-python
```

### Compilation is slower than eager

**This is normal for:**
- Small models
- FP16 workloads
- Memory-bound operations

**Try:**
1. Larger batch sizes
2. FP32 (benefits from TF32)
3. Profile with `TORCH_LOGS="+inductor"` to see which backend is selected

## Advanced Configuration

### Enable Specific Operations Only

```python
# Only use CUTLASS for specific ops
torch._inductor.config.cuda.cutlass_enabled_ops = "mm,addmm,bmm"
```

### Adjust Profiling

```python
# Limit number of CUTLASS configs to try
torch._inductor.config.cuda.cutlass_max_profiling_configs = 10

# Control swizzle options
torch._inductor.config.cuda.cutlass_max_profiling_swizzle_options = [1, 2, 4]
```

### Check Which Backend Was Selected

```bash
TORCH_LOGS="+inductor" python your_script.py 2>&1 | grep -i cutlass
```

Look for lines like:
```
AUTOTUNE mm(64x2048, 2048x4096)
  cutlass_mm_1 0.0123 ms 100.0%  ✅ CUTLASS was fastest
  triton_mm_4  0.0156 ms 78.8%
  mm           0.0178 ms 69.1%   (ATen baseline)
```

## Summary

- ✅ CUTLASS is **working** when installed system-wide
- ✅ Use `mode='max-autotune'` to enable it
- ⚠️  Performance gains depend on workload (compute vs memory-bound)
- 📊 Profile to verify which backend is selected
- 🎯 Best for: large FP32 matmuls, compute-bound operations

## References

- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)
- [PyTorch Compiler Documentation](https://pytorch.org/docs/stable/torch.compiler.html)
- [TorchInductor Configuration](https://pytorch.org/docs/stable/generated/torch._inductor.config.html)

---

**Last Updated:** October 28, 2025
