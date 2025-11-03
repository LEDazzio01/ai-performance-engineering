# Chapter 13: PyTorch Profiling and Optimization

## Overview

PyTorch provides powerful profiling tools to identify bottlenecks in training and inference. This chapter teaches you how to use the PyTorch profiler, analyze memory usage, optimize autograd, and leverage advanced features like compiled autograd and FSDP.

## Learning Objectives

After completing this chapter, you can:

- ✅ Profile PyTorch code to identify CPU and GPU bottlenecks
- ✅ Analyze memory usage and eliminate memory leaks
- ✅ Use compiled autograd for 1.5-2x backward pass speedup
- ✅ Implement custom autograd functions for specialized operations
- ✅ Apply FSDP (Fully Sharded Data Parallel) for large model training
- ✅ Optimize DataLoader and mixed precision training

## Prerequisites

**Previous chapters**:
- [Chapter 1: Performance Basics](../ch1/README.md) - profiling fundamentals
- [Chapter 4: Multi-GPU](../ch4/README.md) - distributed training

**Required**: PyTorch 2.0+, familiarity with PyTorch training loops

## PyTorch Profiling Tools

### Built-in Profiler

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # Your training/inference code
    output = model(input)
    loss = criterion(output, target)
    loss.backward()

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export Chrome trace
prof.export_chrome_trace("trace.json")
```

---

## Examples

### 1. `memory_profiling.py` - Memory Analysis

**Purpose**: Identify memory leaks and optimize memory usage.

#### Memory Profiling Basics

```python
import torch
from torch.profiler import profile, ProfilerActivity

def profile_memory(model, input):
    torch.cuda.reset_peak_memory_stats()
    
    with profile(
        activities=[ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True
    ) as prof:
        output = model(input)
        loss = output.sum()
        loss.backward()
    
    # Memory summary
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=10
    ))
    
    # Peak memory
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Peak memory: {peak_memory:.2f} GB")
    
    return prof

# Example
model = torch.nn.Transformer(d_model=1024, nhead=16).cuda()
input = torch.randn(128, 32, 1024, device='cuda')

prof = profile_memory(model, input)
```

#### Common Memory Issues

```python
# Issue 1: Not releasing intermediate tensors
def bad_memory_pattern():
    results = []
    for i in range(1000):
        x = torch.randn(1000, 1000, device='cuda')
        y = expensive_computation(x)
        results.append(y)  # Keeps all tensors in memory!
    return results

# Fix: Process and release
def good_memory_pattern():
    for i in range(1000):
        x = torch.randn(1000, 1000, device='cuda')
        y = expensive_computation(x)
        process_and_save(y)  # Process immediately, don't accumulate
        del y  # Explicit delete (though not always necessary)

# Issue 2: Gradient accumulation without context manager
def bad_grad_accum(model, data_loader):
    for batch in data_loader:
        output = model(batch)
        loss = criterion(output)
        loss.backward()  # Accumulates gradients AND computation graph!

# Fix: Use no_grad or detach
def good_grad_accum(model, data_loader):
    with torch.no_grad():  # Don't build computation graph
        for batch in data_loader:
            output = model(batch)
            loss = criterion(output)
    loss.backward()  # Only final backward
```

**How to run**:
```bash
python3 memory_profiling.py
```

**Expected output**:
```
-----------  ------------  ------------  ------------  
Name         CPU time      CUDA time     Memory Usage  
-----------  ------------  ------------  ------------  
aten::addmm  12.5 ms       11.2 ms       2.1 GB        
aten::mul    8.3 ms        7.1 ms        1.5 GB        
...
-----------  ------------  ------------  ------------  

Peak memory: 8.4 GB
```

---

### 2. `compiled_autograd.py` - Compiled Autograd (PyTorch 2.0+)

**Purpose**: Use compiled autograd for 1.5-2x faster backward pass.

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
    loss.backward()  # Standard backward

# Compiled autograd (optimized)
def train_compiled_autograd(model, input, target):
    # Enable compiled autograd
    torch._dynamo.config.optimize_ddp = False
    compiled_model = torch.compile(model, mode='reduce-overhead')
    
    output = compiled_model(input)
    loss = criterion(output, target)
    loss.backward()  # Compiled backward!
```

**Benchmark**:

```python
import time

model = YourModel().cuda()
input = torch.randn(128, 3, 224, 224, device='cuda')
target = torch.randint(0, 1000, (128,), device='cuda')

# Warmup
for _ in range(10):
    train_regular(model, input, target)

# Benchmark regular
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    train_regular(model, input, target)
torch.cuda.synchronize()
regular_time = time.time() - start

# Benchmark compiled
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    train_compiled_autograd(model, input, target)
torch.cuda.synchronize()
compiled_time = time.time() - start

print(f"Regular: {regular_time:.2f}s")
print(f"Compiled: {compiled_time:.2f}s")
print(f"Speedup: {regular_time / compiled_time:.2f}x")
```

**Expected speedup**: **1.5-2x** for transformer models.

**How to run**:
```bash
python3 compiled_autograd.py
```

---

### 3. `custom_allocator.py` - Custom Memory Allocator

**Purpose**: Implement custom allocator for specialized memory management.

```python
import torch

class CustomCachingAllocator:
    """Pool allocator that reduces cudaMalloc calls."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.pools = {}  # size -> list of free blocks
        self.allocated = {}  # ptr -> (size, in_use)
    
    def allocate(self, size):
        # Round up to power of 2
        size = 2 ** (size - 1).bit_length()
        
        # Check pool for free block
        if size in self.pools and self.pools[size]:
            ptr = self.pools[size].pop()
            self.allocated[ptr] = (size, True)
            return ptr
        
        # Allocate new block
        ptr = torch.cuda.caching_allocator_alloc(size, device=self.device)
        self.allocated[ptr] = (size, True)
        return ptr
    
    def free(self, ptr):
        if ptr in self.allocated:
            size, _ = self.allocated[ptr]
            self.allocated[ptr] = (size, False)
            
            # Return to pool
            if size not in self.pools:
                self.pools[size] = []
            self.pools[size].append(ptr)

# Usage
allocator = CustomCachingAllocator()

for _ in range(1000):
    ptr = allocator.allocate(1024 * 1024)  # 1 MB
    # Use memory...
    allocator.free(ptr)

# Much faster than 1000 cudaMalloc/cudaFree calls!
```

**How to run**:
```bash
python3 custom_allocator.py
```

---

### 4. `fsdp_example.py` - Fully Sharded Data Parallel

**Purpose**: Train large models that don't fit on single GPU.

**What is FSDP?**
- Shards model parameters, gradients, and optimizer states across GPUs
- Each GPU only stores 1/N of the model
- Enables training models 8x larger than single-GPU memory

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload, MixedPrecision

# Initialize distributed
dist.init_process_group("nccl")
rank = dist.get_rank()
device = torch.device(f"cuda:{rank}")

# Create large model
model = VeryLargeModel().to(device)  # e.g., 70B parameters

# Wrap with FSDP
model = FSDP(
    model,
    # Shard across 8 GPUs
    device_id=device,
    
    # Mixed precision for memory savings
    mixed_precision=MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
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
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Memory savings**:
- **8x GPUs**: Train 8x larger model than single GPU
- **With CPU offload**: Train 16-32x larger (slower, but possible)

**How to run**:
```bash
torchrun --nproc_per_node=8 fsdp_example.py
```

---

### 5. `compare_perf.py` - Performance Comparison Tool

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
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return BenchmarkResult(
        name=name,
        mean_time=sum(times) / len(times),
        std_time=torch.tensor(times).std().item(),
        memory_peak=torch.cuda.max_memory_allocated() / (1024**3)
    )

# Example usage
results = []
results.append(benchmark("Baseline", lambda: model(input)))
results.append(benchmark("torch.compile", lambda: compiled_model(input)))
results.append(benchmark("Mixed Precision", lambda: model_amp(input)))

# Print comparison
for r in results:
    print(f"{r.name:20s}: {r.mean_time*1000:6.2f} ms, {r.memory_peak:5.2f} GB")
```

**How to run**:
```bash
python3 compare_perf.py
```

---

### 6. `train_deepseek_coder.py` / `train_deepseek_v3.py` - Real Model Examples

**Purpose**: Profile and optimize real-world large model training.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load DeepSeek Coder (6.7B parameters)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-base",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Profile training step
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    with_modules=True
) as prof:
    # Training step
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()

# Analyze results
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20
))

# Export for visualization
prof.export_chrome_trace("deepseek_coder_trace.json")
```

**How to run**:
```bash
python3 train_deepseek_coder.py
```

**View trace**:
1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Load `deepseek_coder_trace.json`
4. Analyze timeline

---

## Common Optimization Patterns

### 1. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

model = Model().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward in mixed precision
    with autocast():
        output = model(batch['input'])
        loss = criterion(output, batch['target'])
    
    # Scaled backward
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Memory savings**: **~40%** (FP16 vs FP32)  
**Speedup**: **1.5-2x** on Tensor Cores

### 2. Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class ModelWithCheckpointing(nn.Module):
    def forward(self, x):
        # Checkpointed layers (recompute in backward)
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        x = checkpoint(self.layer3, x)
        return x
```

**Memory savings**: **N layers → 1 layer memory** (recomputation cost)  
**Trade-off**: 20-30% slower backward, but enables larger batch sizes

### 3. Efficient DataLoader

```python
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    num_workers=8,           # Parallel loading
    pin_memory=True,         # Faster H2D transfers
    persistent_workers=True, # Keep workers alive
    prefetch_factor=2,       # Prefetch batches
)
```

---

## How to Run All Examples

```bash
cd ch13

# Install dependencies
pip install -r requirements.txt

# Memory profiling
python3 memory_profiling.py

# Compiled autograd
python3 compiled_autograd.py

# Custom allocator
python3 custom_allocator.py

# FSDP (requires 8 GPUs)
torchrun --nproc_per_node=8 fsdp_example.py

# Performance comparison
python3 compare_perf.py

# Real model profiling
python3 train_deepseek_coder.py

# View traces in Chrome
# chrome://tracing → Load *.json file
```

---

## Key Takeaways

1. **Profile first, optimize second**: Use PyTorch profiler to identify actual bottlenecks.

2. **Memory is often the limit**: Profile memory to find leaks and optimize usage before scaling up.

3. **Compiled autograd gives free speedup**: 1.5-2x backward pass with `torch.compile`.

4. **FSDP for large models**: Train 8x larger models by sharding across GPUs.

5. **Mixed precision is essential**: FP16/BF16 saves 40% memory and gives 1.5-2x speedup on Tensor Cores.

6. **Gradient checkpointing trades time for memory**: Recompute activations in backward to save memory.

7. **Chrome trace is your friend**: Visual timeline shows gaps, overlaps, and bottlenecks clearly.

---

## Common Pitfalls

### Pitfall 1: Profiling Without Warmup
**Problem**: First iterations include compilation, autotuning → Skewed results.

**Solution**: Always warmup 10-20 iterations before profiling.

### Pitfall 2: Accumulating Tensors in Lists
**Problem**: `results.append(tensor)` keeps entire computation graph!

**Solution**: Detach or convert to Python: `results.append(tensor.detach().cpu())`.

### Pitfall 3: Not Using `torch.no_grad()` for Inference
**Problem**: Building computation graph during inference → Wasted memory.

**Solution**: Always wrap inference with `torch.no_grad()` or `model.eval()`.

### Pitfall 4: Forgetting `optimizer.zero_grad()`
**Problem**: Gradients accumulate indefinitely → Memory leak!

**Solution**: Call `optimizer.zero_grad()` at start of each iteration.

### Pitfall 5: Profiling with Too Few Iterations
**Problem**: High variance in timing measurements.

**Solution**: Profile 100+ iterations and report mean ± std.

---

## Next Steps

**Compiler optimizations** → [Chapter 14: torch.compile and Triton](../ch14/README.md)

Learn about:
- torch.compile for automatic optimization
- Writing custom Triton kernels
- TMA in Triton (when it works!)
- Compiler modes and trade-offs

**Back to CUDA** → [Chapter 10: Tensor Cores](../ch10/README.md)

---

## Additional Resources

- **PyTorch Profiler**: [Official Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- **FSDP Documentation**: [Fully Sharded Data Parallel](https://pytorch.org/docs/stable/fsdp.html)
- **Mixed Precision**: [Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- **Compiled Autograd**: [PyTorch 2.0 Features](https://pytorch.org/get-started/pytorch-2.0/)

---

**Chapter Status**: ✅ Complete  
**Last Updated**: November 3, 2025  
**Tested On**: 8x NVIDIA B200 GPUs, PyTorch 2.9, CUDA 13.0

