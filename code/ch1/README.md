# Chapter 1: Performance Basics

## Overview

This chapter establishes the foundation for all performance optimization work. Learn to profile code, identify bottlenecks, and apply fundamental optimizations that deliver 5-10x speedups. These techniques are universally applicable and form the basis for more advanced optimizations in later chapters.

## Learning Objectives

After completing this chapter, you can:

- [OK] Profile Python/PyTorch code to identify performance bottlenecks
- [OK] Measure goodput (useful compute vs total time) to quantify efficiency
- [OK] Apply memory management optimizations (pinned memory, preallocated buffers)
- [OK] Use batched operations to improve GPU utilization
- [OK] Leverage CUDA Graphs to reduce kernel launch overhead
- [OK] Understand when and how to apply fundamental optimizations

## Prerequisites

**None** - This is the foundation chapter. Start here!

**Hardware**: NVIDIA GPU

**Software**: PyTorch [file]+, CUDA [file]+

## Examples

### 1. Baseline Implementation

**Purpose**: Establish baseline performance measurement methodology.

**What it demonstrates**:
- Basic training loop structure
- Common inefficiencies in naive implementations
- Benchmark protocol integration

**How to run**:
```bash
python3 [baseline script]
```

**Expected output**:
```
Baseline: Performance Basics
======================================================================
Average time: [file] ms
Median: [file] ms
Std: [file] ms
```

Typical baseline: **40-60% goodput** (significant overhead from tensor creation and transfers)

---

### 2. Optimized Implementations

**Purpose**: Apply fundamental optimizations individually to measure impact.

This chapter demonstrates optimizations through separate benchmark files:

#### 2a. Pinned Memory Optimization

**Problem**: Unpinned memory requires CPU staging buffer for H2D transfers.

**Solution**: Use pinned memory for faster CPU-GPU transfers:
```python
host_data = [file](32, 256, pin_memory=True)
[file]_(host_data, non_blocking=True)  # Non-blocking copy
```

**Impact**: 2-6x faster H2D transfers (varies by system)

**How to run**:
```bash
python3 [pinned memory script]
```

#### 2b. Larger Batch Size Optimization

**Problem**: Small batches (32) underutilize GPU (low MFLOPs).

**Solution**: Increase batch size to saturate compute:
```python
batch_size = 256  # vs baseline 32
```

**Impact**: 87 MFLOPs → 1000+ MFLOPs (10x+ GEMM efficiency)

**How to run**:
```bash
python3 [batch size script]
```

#### 2c. CUDA Graphs Optimization

**Problem**: Each kernel launch has ~5-20μs overhead. Small kernels spend more time launching than computing!

**Solution**: Capture static computation graph and replay:
```python
graph = [file].CUDAGraph()
with [file].graph(graph):
    output = model(input)
[file]()  # Much faster than re-launching
```

**Impact**: ~50-70% reduction in launch overhead

**Key optimizations also included**:
- Preallocated device buffers (eliminates tensor creation overhead)
- Pinned memory for faster transfers

**How to run**:
```bash
python3 [CUDA graphs script]
```

**Compare all optimizations**:
```bash
python3 [compare script]  # Compares baseline vs all optimized variants
```

**Expected overall speedup**: **2-5x** (varies by workload and hardware)

---

### 3. CUDA GEMM Examples - Batched GEMM Optimization

**Purpose**: Demonstrate importance of batched operations at CUDA level.

**Problem observed in profiling**: Training loop launched 40 separate GEMMs sequentially:
- Each launch: ~10μs overhead
- Total overhead: 400μs per batch
- Poor kernel fusion opportunities

**Solution**: Use cuBLAS batched GEMM API:
```cpp
cublasSgemmBatched(handle, ..., batch_count);
```

**How to run**:
```bash
cd ch1
make
# Run the compiled binaries (architecture suffix added automatically)
```

**Expected output**:
```
Individual GEMMs: XXX ms
Batched GEMM:     YYY ms
Speedup:          [file]
```

**Typical speedup**: **20-40x** (more dramatic for small matrices)

**Key insight**: This is why PyTorch automatically batches operations internally!

---

### 4. Roofline Performance Model

**Purpose**: Implement roofline analysis to classify kernels as compute-bound or memory-bound.

**What it demonstrates**:
- Calculating arithmetic intensity (FLOP/Byte) for different operations
- Plotting kernels on the roofline model for NVIDIA GPUs
- Identifying whether optimizations should target compute or memory bandwidth
- Comparing vector operations (memory-bound) vs matrix operations (compute-bound)

**Key concepts**:
- **Roofline model**: Performance ceiling defined by either compute peak or memory bandwidth
- **Ridge point**: Arithmetic intensity where compute and bandwidth ceilings intersect
- **Example specs**: Modern NVIDIA GPUs achieve high TFLOPS and memory bandwidth
- **Optimization strategy**: Memory-bound kernels need better data reuse; compute-bound kernels need better instruction mix

**How to run**:
```bash
python3 [roofline analysis script]
```

**Expected output**:
```
Vector Add:
  AI: [file] FLOP/Byte
  Achieved: [file] TFLOPS
  Memory-bound (AI << 250)

Matrix Multiply:
  AI: [file] FLOP/Byte
  Achieved: [file] TFLOPS
  Compute-bound (AI > 250)

Roofline plot saved to [file]
```

---

### Chapter profiling

Chapter profiling is handled by the compare script. Run it from the project root:

```bash
python3 -c "from [file] import profile; profile()"
```

Or run benchmarks using the unified entry point:
```bash
python [benchmark script] --chapter 1
```

**Key insight**: Operations below the ridge point are limited by memory bandwidth, not compute!

---

### 5. Arithmetic Intensity Optimization - Kernel Optimization Strategies

**Purpose**: Show kernel optimization techniques and their impact on arithmetic intensity.

**Implementation**: Baseline and optimized CUDA kernels demonstrating progressive optimizations

**What it demonstrates**:
- **Baseline**: Simple element-wise kernel
- **Loop unrolling**: Reduce branch overhead, expose ILP
- **Vectorized loads**: Use `float4` to load 128 bits at once
- **Increased FLOPs**: Add useful work to improve AI
- **Kernel fusion**: Combine multiple passes to eliminate memory traffic

**Performance progression**:
```
Baseline:     125 GB/s,  AI = [file] FLOP/Byte
Unrolled:     145 GB/s,  AI = [file] FLOP/Byte (better utilization)
Vectorized:   245 GB/s,  AI = [file] FLOP/Byte (coalescing + bandwidth)
More FLOPs:   280 GB/s,  AI = [file] FLOP/Byte (6x better AI!)
Fused:        420 GB/s,  AI = [file] FLOP/Byte (12x better AI!)
```

**How to run**:
```bash
make
# Run the compiled binaries (add appropriate architecture suffix)
# Example: ```

**Expected output**:
```
Arithmetic Intensity Optimization Demo (N = 10M elements)

Baseline kernel:
  Time: [file] ms, Bandwidth: [file] GB/s, AI: [file] FLOP/Byte

Unrolled kernel:  
  Time: [file] ms, Bandwidth: [file] GB/s, AI: [file] FLOP/Byte

Vectorized kernel:
  Time: [file] ms, Bandwidth: [file] GB/s, AI: [file] FLOP/Byte

Optimized kernel (more FLOPs):
  Time: [file] ms, Bandwidth: [file] GB/s, AI: [file] FLOP/Byte

Fused kernel:
  Time: [file] ms, Bandwidth: [file] GB/s, AI: [file] FLOP/Byte
  Overall speedup: [file]
```

**Key insight**: Increasing arithmetic intensity (more FLOPs per byte) reduces memory bottlenecks and improves performance!

---

## Performance Analysis

### Profiling Your Own Code

Use the common profiling infrastructure:

```bash
# Profile Python examples
../.[executable]/profiling/[file] [baseline script]
../.[executable]/profiling/[file] [optimized script]

# View timeline in Nsight Systems
nsys-ui [profile output file]
```

**What to look for**:
- ERROR: Long CPU gaps between GPU kernels → Add async operations
- ERROR: Many small kernel launches → Batch or fuse operations
- ERROR: `aten::empty_strided` taking significant time → Preallocate buffers
- [OK] GPU utilization > 80% → Good!

### Expected Performance Improvements

| Optimization | Baseline → Optimized | Speedup |
|--------------|---------------------|---------|
| Preallocated buffers | 210ms overhead → 0ms | ~2x |
| Pinned memory | System dependent | 2-6x |
| CUDA Graphs | 5-20μs/launch → <1μs | [file]-2x |
| Larger batches | 87 MFLOPs → 1000+ | 10x+ |
| **Combined** | **Overall end-to-end** | **5-10x** |

*Your results may vary by hardware.*

---

## Baseline/Optimized Example Pairs

All examples follow the baseline/optimized pattern and integrate with the benchmarking framework:

### Available Pairs

1. **Coalescing** - Baseline and optimized implementations
   - Demonstrates coalesced vs uncoalesced memory access patterns
   - Shows bandwidth improvements from proper memory access

2. **Double Buffering** - Baseline and optimized implementations
   - Overlaps memory transfer and computation using CUDA streams
   - Demonstrates latency hiding through async operations

**Run comparisons:**
```bash
python3 [compare script]  # Compares all baseline/optimized pairs
```

---

## How to Run All Examples

```bash
cd ch1

# Install dependencies
pip install -r [file]

# Run Python examples
python3 [baseline script]                        # Baseline
python3 [pinned memory script]                   # Pinned memory optimization
python3 [batch size script]                      # Larger batch size
python3 [CUDA graphs script]                     # CUDA Graphs optimization
python3 [roofline script]                        # Roofline model

# Run baseline/optimized comparisons
python3 [compare script]                         # Compare all pairs

# Build and run CUDA examples
make
# Run compiled binaries (architecture suffix added automatically)

# Profile examples (optional)
../.[executable]/profiling/[file] [baseline script]
../.[executable]/profiling/[file] [optimized script]
../.[executable]/profiling/[file] [CUDA binary] ch1_ai
```

---

## Key Takeaways

1. **Always profile first**: Don't optimize blindly. Use profilers to identify actual bottlenecks.

2. **Memory management matters**: Preallocating buffers and using pinned memory can give 2-6x speedups with minimal code changes.

3. **Batch operations**: GPUs thrive on parallelism. Batching operations reduces overhead and improves efficiency dramatically (10x+ in many cases).

4. **Launch overhead is real**: For small operations, kernel launch overhead dominates. CUDA Graphs and batching mitigate this.

5. **Compound improvements**: Individual optimizations multiply. A 2x + 2x + [file] → 6x combined speedup.

6. **Low-hanging fruit**: These optimizations require minimal code changes but deliver major improvements. Always apply them first!

---

## Common Pitfalls

### Pitfall 1: Over-batching
**Problem**: Batch size too large → OOM (out of memory) errors.

**Solution**: Find sweet spot using batch size sweep. Typical range: 64-512 for modern NVIDIA GPUs.

### Pitfall 2: CUDA Graphs with Dynamic Shapes
**Problem**: CUDA Graphs require static shapes. Dynamic models will fail or show no speedup.

**Solution**: Only use graphs for static portions of your model. Prefill/decode in inference are good candidates.

### Pitfall 3: Measuring Without Synchronization
**Problem**: CUDA operations are async. `[file]()` without `[file].synchronize()` measures queue time, not execution time!

**Solution**: Always synchronize before timing:
```python
[file].synchronize()
start = [file]()
model(input)
[file].synchronize()  # Critical!
elapsed = [file]() - start
```

### Pitfall 4: Cold Start Measurements
**Problem**: First few iterations include GPU warmup, driver overhead, cuDNN autotuning.

**Solution**: Always warmup (10-20 iterations) before benchmarking.

---

## Next Steps

**Ready for more?** → [Chapter 2: GPU Hardware Architecture](.[executable]/[file])

Learn about:
- NVIDIA GPU hardware architecture
- Memory hierarchy
- NVLink and interconnects
- How hardware architecture informs optimization strategy

**Want to dive deeper into profiling?** → [Chapter 13: PyTorch Profiling](.[executable]/[file])

---

## Additional Resources

- **Official Docs**: [PyTorch Performance Tuning Guide](https://[file]/tutorials/recipes/recipes/[file])
- **cuBLAS Documentation**: [CUDA Toolkit Docs - cuBLAS](https://[file].com/cuda/cublas/)
- **CUDA Graphs**: [CUDA Programming Guide - Graphs](https://[file].com/cuda/cuda-c-programming-guide/[file]#cuda-graphs)

---

**Chapter Status**: [OK] Complete
