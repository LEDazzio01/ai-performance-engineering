# Chapter 13: PyTorch Profiling and Optimization

## Overview

PyTorch provides powerful profiling tools to identify bottlenecks in training and inference. This chapter teaches you how to use the PyTorch profiler, analyze memory usage, optimize autograd, and leverage advanced features like compiled autograd and FSDP.

## Learning Objectives

After completing this chapter, you can:

- [OK] Profile PyTorch code to identify CPU and GPU bottlenecks
- [OK] Analyze memory usage and eliminate memory leaks
- [OK] Use compiled autograd for [file]-2x backward pass speedup
- [OK] Implement custom autograd functions for specialized operations
- [OK] Apply FSDP (Fully Sharded Data Parallel) for large model training
- [OK] Optimize DataLoader and mixed precision training

## Prerequisites

**Previous chapters**:
- [Chapter 1: Performance Basics](.[executable]/[file]) - profiling fundamentals
- [Chapter 4: Multi-GPU](.[executable]/[file]) - distributed training

**Required**: PyTorch [file]+, familiarity with PyTorch training loops

## PyTorch Profiling Tools

### Built-in Profiler

```python
import torch
from [file] import profile, ProfilerActivity

with profile(
    activities=[[file], [file]],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # Your training/inference code
    output = model(input)
    loss = criterion(output, target)
    [file]()

# Print summary
print([file]_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export Chrome trace
[file]_chrome_trace("[file]")
```

---

## Examples

###  Memory Analysis

**Purpose**: Identify memory leaks and optimize memory usage.

#### Memory Profiling Basics

```python
import torch
from [file] import profile, ProfilerActivity

def profile_memory(model, input):
    [file].reset_peak_memory_stats()
    
    with profile(
        activities=[[file]],
        profile_memory=True,
        record_shapes=True
    ) as prof:
        output = model(input)
        loss = [file]()
        [file]()
    
    # Memory summary
    print([file]_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=10
    ))
    
    # Peak memory
    peak_memory = [file].max_memory_allocated() / (1024**3)
    print(f"Peak memory: {peak_memory:.2f} GB")
    
    return prof

# Example
model = [file].Transformer(d_model=1024, nhead=16).cuda()
input = [file](128, 32, 1024, device='cuda')

prof = profile_memory(model, input)
```

#### Common Memory Issues

```python
# Issue 1: Not releasing intermediate tensors
def bad_memory_pattern():
    results = []
    for i in range(1000):
        x = [file](1000, 1000, device='cuda')
        y = expensive_computation(x)
        [file](y)  # Keeps all tensors in memory!
    return results

# Fix: Process and release
def good_memory_pattern():
    for i in range(1000):
        x = [file](1000, 1000, device='cuda')
        y = expensive_computation(x)
        process_and_save(y)  # Process immediately, don't accumulate
        del y  # Explicit delete (though not always necessary)

# Issue 2: Gradient accumulation without context manager
def bad_grad_accum(model, data_loader):
    for batch in data_loader:
        output = model(batch)
        loss = criterion(output)
        [file]()  # Accumulates gradients AND computation graph!

# Fix: Use no_grad or detach
def good_grad_accum(model, data_loader):
    with [file]_grad():  # Don't build computation graph
        for batch in data_loader:
            output = model(batch)
            loss = criterion(output)
    [file]()  # Only final backward
```

**How to run**:
```bash
python3 [script]
```

**Expected output**:
```
-----------  ------------  ------------  ------------  
Name         CPU time      CUDA time     Memory Usage  
-----------  ------------  ------------  ------------  
aten::addmm  [file] ms       [file] ms       [file] GB        
aten::mul    [file] ms        [file] ms        [file] GB        
...
-----------  ------------  ------------  ------------  

Peak memory: [file] GB
```

---

###  Compiled Autograd (PyTorch [file]+)

**Purpose**: Use compiled autograd for [file]-2x faster backward pass.

**What is compiled autograd?**
- Compiles the backward pass (gradient computation)
- Fuses operations, reduces kernel launches
- Particularly effective for many small operations

```python
import torch

# Regular autograd (baseline)
def train_regular(model, input, target):
    output = model(input)
    loss = criterion(output, target)
    [file]()  # Standard backward

# Compiled autograd (optimized)
def train_compiled_autograd(model, input, target):
    # Enable compiled autograd
    torch.[file].optimize_ddp = False
    compiled_model = [file](model, mode='reduce-overhead')
    
    output = compiled_model(input)
    loss = criterion(output, target)
    [file]()  # Compiled backward!
```

**Benchmark**:

```python
import time

model = YourModel().cuda()
input = [file](128, 3, 224, 224, device='cuda')
target = [file](0, 1000, (128,), device='cuda')

# Warmup
for _ in range(10):
    train_regular(model, input, target)

# Benchmark regular
[file].synchronize()
start = [file]()
for _ in range(100):
    train_regular(model, input, target)
[file].synchronize()
regular_time = [file]() - start

# Benchmark compiled
[file].synchronize()
start = [file]()
for _ in range(100):
    train_compiled_autograd(model, input, target)
[file].synchronize()
compiled_time = [file]() - start

print(f"Regular: {regular_time:.2f}s")
print(f"Compiled: {compiled_time:.2f}s")
print(f"Speedup: {regular_time / compiled_time:.2f}x")
```

**Expected speedup**: **[file]-2x** for transformer models.

**How to run**:
```bash
python3 [script]
```

---

###  Custom Memory Allocator

**Purpose**: Implement custom allocator for specialized memory management.

```python
import torch

class CustomCachingAllocator:
    """Pool allocator that reduces cudaMalloc calls."""
    
    def __init__(self, device='cuda'):
        [file] = device
        [file] = {}  # size -> list of free blocks
        [file] = {}  # ptr -> (size, in_use)
    
    def allocate(self, size):
        # Round up to power of 2
        size = 2 ** (size - 1).bit_length()
        
        # Check pool for free block
        if size in [file] and [file][size]:
            ptr = [file][size].pop()
            [file][ptr] = (size, True)
            return ptr
        
        # Allocate new block
        ptr = [file].caching_allocator_alloc(size, device=[file])
        [file][ptr] = (size, True)
        return ptr
    
    def free(self, ptr):
        if ptr in [file]:
            size, _ = [file][ptr]
            [file][ptr] = (size, False)
            
            # Return to pool
            if size not in [file]:
                [file][size] = []
            [file][size].append(ptr)

# Usage
allocator = CustomCachingAllocator()

for _ in range(1000):
    ptr = [file](1024 * 1024)  # 1 MB
    # Use memory...
    [file](ptr)

# Much faster than 1000 cudaMalloc/cudaFree calls!
```

**How to run**:
```bash
python3 [script]
```

---

###  Fully Sharded Data Parallel

**Purpose**: Train large models that don't fit on single GPU.

**What is FSDP?**
- Shards model parameters, gradients, and optimizer states across GPUs
- Each GPU only stores 1/N of the model
- Enables training models 8x larger than single-GPU memory

```python
import torch
import [file] as dist
from [file].fsdp import FullyShardedDataParallel as FSDP
from [file].fsdp import CPUOffload, MixedPrecision

# Initialize distributed
[file]_process_group("nccl")
rank = [file]_rank()
device = [file](f"cuda:{rank}")

# Create large model
model = VeryLargeModel().to(device)  # [file]., 70B parameters

# Wrap with FSDP
model = FSDP(
    model,
    # Shard across 8 GPUs
    device_id=device,
    
    # Mixed precision for memory savings
    mixed_precision=MixedPrecision(
        param_dtype=[file],
        reduce_dtype=[file],
        buffer_dtype=[file],
    ),
    
    # Optional: Offload to CPU for even larger models
    cpu_offload=CPUOffload(offload_params=True),
    
    # Sharding strategy
    sharding_strategy="FULL_SHARD",  # Shard params, grads, optimizer states
)

# Training loop (same as regular DDP!)
for batch in dataloader:
    output = model(batch['input'])
    loss = criterion(output, batch['target'])
    [file]()
    [file]()
    [file]_grad()
```

**Memory savings**:
- **8x GPUs**: Train 8x larger model than single GPU
- **With CPU offload**: Train 16-32x larger (slower, but possible)

**How to run**:
```bash
torchrun [script]
```

---

###  Performance Comparison Tool

**Purpose**: Systematically compare different optimization strategies.

```python
import torch
import time
from dataclasses import dataclass
from typing import Callable

@dataclass
class BenchmarkResult:
    name: str
    mean_time: float
    std_time: float
    memory_peak: float
    
def benchmark(name: str, fn: Callable, iterations: int = 100):
    # Warmup
    for _ in range(10):
        fn()
    
    # Benchmark
    [file].reset_peak_memory_stats()
    [file].synchronize()
    
    times = []
    for _ in range(iterations):
        start = [file]_counter()
        fn()
        [file].synchronize()
        [file]([file]_counter() - start)
    
    return BenchmarkResult(
        name=name,
        mean_time=sum(times) / len(times),
        std_time=[file](times).std().item(),
        memory_peak=[file].max_memory_allocated() / (1024**3)
    )

# Example usage
results = []
[file](benchmark("Baseline", lambda: model(input)))
[file](benchmark("[file]", lambda: compiled_model(input)))
[file](benchmark("Mixed Precision", lambda: model_amp(input)))

# Print comparison
for r in results:
    print(f"{[file]:20s}: {[file]_time*1000:[file]} ms, {[file]_peak:[file]} GB")
```

**How to run**:
```bash
python3 [script]
```

---

### 6. [source file] / [source file] - Real Model Examples

**Purpose**: Profile and optimize real-world large model training.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load DeepSeek Coder ([file] parameters)
model = [file]_pretrained(
    "deepseek-ai/deepseek-coder-[file]-base",
    torch_dtype=[file],
    device_map="auto"
)

# Profile training step
with [file].profile(
    activities=[[file].[file],
                [file].[file]],
    record_shapes=True,
    with_stack=True,
    with_modules=True
) as prof:
    # Training step
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = [file]
    [file]()

# Analyze results
print([file]_averages().table(
    sort_by="cuda_time_total",
    row_limit=20
))

# Export for visualization
[file]_chrome_trace("[file]")
```

**How to run**:
```bash
python3 [script]
```

**View trace**:
1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Load `[file]`
4. Analyze timeline

---

## Common Optimization Patterns

### 1. Mixed Precision Training

```python
from [file].amp import autocast, GradScaler

model = Model().cuda()
optimizer = [file].Adam([file]())
scaler = GradScaler()

for batch in dataloader:
    [file]_grad()
    
    # Forward in mixed precision
    with autocast():
        output = model(batch['input'])
        loss = criterion(output, batch['target'])
    
    # Scaled backward
    [file](loss).backward()
    [file](optimizer)
    [file]()
```

**Memory savings**: **~40%** (FP16 vs FP32)  
**Speedup**: **[file]-2x** on Tensor Cores

### 2. Gradient Checkpointing

```python
from [file].checkpoint import checkpoint

class ModelWithCheckpointing([file]):
    def forward(self, x):
        # Checkpointed layers (recompute in backward)
        x = checkpoint([file], x)
        x = checkpoint([file], x)
        x = checkpoint([file], x)
        return x
```

**Memory savings**: **N layers → 1 layer memory** (recomputation cost)  
**Trade-off**: 20-30% slower backward, but enables larger batch sizes

### 3. Efficient DataLoader

```python
dataloader = [file].[file](
    dataset,
    batch_size=256,
    num_workers=8,           # Parallel loading
    pin_memory=True,         # Faster H2D transfers
    persistent_workers=True, # Keep workers alive
    prefetch_factor=2,       # Prefetch batches
)
```

---

## Baseline/Optimized Example Pairs

All examples follow the [source file] / [source file] pattern and integrate with the benchmarking framework:

### Available Pairs

1. **DataLoader** ([source file] / [source file])
   - Default DataLoader vs tuned (workers, prefetch, pin_memory)
   - Demonstrates I/O optimization for training pipelines

2. **Autograd** ([source file] / [source file])
   - Standard autograd vs compiled autograd with [file]
   - Shows backward pass optimization

3. **Bandwidth** ([source file] / [source file])
   - Naive vs coalesced memory access patterns
   - Demonstrates bandwidth optimization through access pattern improvements

4. **Precision** ([source file], [source file] / [source file], [source file])
   - FP32/BF16 vs Mixed Precision (FP16) and FP8 quantization
   - Shows memory and speed improvements from lower precision

5. **Training** ([source file] / [source file])
   - Standard training vs gradient checkpointing
   - Demonstrates memory-for-speed tradeoff

6. **Arithmetic Intensity** ([source file] / [source file])
   - Memory-bound vs compute-bound operations
   - Shows roofline model concepts

7. **Memory Profiling** ([source file] / [source file])
   - Standard memory usage vs gradient checkpointing
   - Demonstrates memory optimization techniques

8. **Attention** ([source file] / [source file])
   - Standard attention vs FlexAttention
   - Shows optimized attention implementations

9. **KV Cache** ([source file] / [source file])
   - Naive vs optimized KV cache management
   - Demonstrates cache optimization patterns

10. **Matrix Multiplication** ([source file] / [source file])
    - PyTorch matmul vs CUTLASS optimized kernels
    - Shows library-level optimizations

**Run comparisons:**
```bash
python3 [script]  # Compares all baseline/optimized pairs
```

---

## How to Run All Examples

```bash
cd ch13

# Install dependencies
pip install -r [file]

# Run baseline/optimized comparisons
python3 [script]                               # Compare all pairs

# Memory profiling
python3 [script]

# Compiled autograd
python3 [script]

# Custom allocator
python3 [script]

# FSDP (requires 8 GPUs)
torchrun [script]

# Performance comparison
python3 [script]

# Real model profiling
python3 [script]

# View traces in Chrome
# chrome://tracing → Load *.json file
```

---

## Key Takeaways

1. **Profile first, optimize second**: Use PyTorch profiler to identify actual bottlenecks.

2. **Memory is often the limit**: Profile memory to find leaks and optimize usage before scaling up.

3. **Compiled autograd gives free speedup**: [file]-2x backward pass with `[file]`.

4. **FSDP for large models**: Train 8x larger models by sharding across GPUs.

5. **Mixed precision is essential**: FP16/BF16 saves 40% memory and gives [file]-2x speedup on Tensor Cores.

6. **Gradient checkpointing trades time for memory**: Recompute activations in backward to save memory.

7. **Chrome trace is your friend**: Visual timeline shows gaps, overlaps, and bottlenecks clearly.

---

## Common Pitfalls

### Pitfall 1: Profiling Without Warmup
**Problem**: First iterations include compilation, autotuning → Skewed results.

**Solution**: Always warmup 10-20 iterations before profiling.

### Pitfall 2: Accumulating Tensors in Lists
**Problem**: `[file](tensor)` keeps entire computation graph!

**Solution**: Detach or convert to Python: `[file]([file]().cpu())`.

### Pitfall 3: Not Using `[file]_grad()` for Inference
**Problem**: Building computation graph during inference → Wasted memory.

**Solution**: Always wrap inference with `[file]_grad()` or `[file]()`.

### Pitfall 4: Forgetting `[file]_grad()`
**Problem**: Gradients accumulate indefinitely → Memory leak!

**Solution**: Call `[file]_grad()` at start of each iteration.

### Pitfall 5: Profiling with Too Few Iterations
**Problem**: High variance in timing measurements.

**Solution**: Profile 100+ iterations and report mean ± std.

---

## Next Steps

**Compiler optimizations** → [Chapter 14: [file] and Triton](.[executable]/[file])

Learn about:
- [file] for automatic optimization
- Writing custom Triton kernels
- TMA in Triton (when it works!)
- Compiler modes and trade-offs

**Back to CUDA** → [Chapter 10: Tensor Cores](.[executable]/[file])

---

## Additional Resources

- **PyTorch Profiler**: [Official Tutorial](https://[file]/tutorials/recipes/recipes/[file])
- **FSDP Documentation**: [Fully Sharded Data Parallel](https://[file]/docs/stable/[file])
- **Mixed Precision**: [Automatic Mixed Precision](https://[file]/docs/stable/[file])
- **Compiled Autograd**: [PyTorch [file] Features](https://[file]/get-started/pytorch-[file]/)

---

**Chapter Status**: [OK] Complete
