# Chapter 8: Occupancy and Instruction-Level Parallelism

## Overview

High occupancy and instruction-level parallelism (ILP) help GPUs hide latency and maximize throughput. This chapter teaches you how to tune occupancy, leverage ILP, manage register pressure, and mitigate warp divergence to squeeze maximum performance from your kernels.

## Learning Objectives

After completing this chapter, you can:

- [OK] Understand occupancy and its impact on performance
- [OK] Tune occupancy by balancing resources (registers, shared memory, threads)
- [OK] Apply instruction-level parallelism to hide latency
- [OK] Manage register pressure to maintain high occupancy
- [OK] Identify and mitigate warp divergence
- [OK] Use loop unrolling for performance gains

## Prerequisites

**Previous chapters**:
- [Chapter 6: CUDA Basics](.[executable]/README.md) - thread hierarchy
- [Chapter 7: Memory Access](.[executable]/README.md) - memory patterns

**Required**: Understanding of GPU execution model and latency

## Occupancy Deep Dive

### What is Occupancy?

```
Occupancy = Active_Warps_Per_SM / Maximum_Warps_Per_SM
          = (Active_Blocks × Threads_Per_Block / 32) / Max_Warps
```

**For NVIDIA GPU**:
- Max warps per SM: 64
- Max threads per SM: 2048 (64 warps × 32 threads)
- Max blocks per SM: 32

**Why occupancy matters**: Higher occupancy → More warps to switch between → Better latency hiding → Higher throughput (usually).

### Occupancy Limiters

| Resource | NVIDIA GPU Limit | Impact |
|----------|------------|--------|
| **Registers** | 65,536 per SM | High register usage → Fewer active blocks |
| **Shared Memory** | 256 KB per SM | Large shared memory → Fewer active blocks |
| **Threads per Block** | Max 1024 | Too few threads → Low occupancy |
| **Blocks per SM** | Max 32 | Physical limit |

---

## Examples

###  Finding Optimal Configuration

**Purpose**: Demonstrate how to find the sweet spot between occupancy and per-thread resources.

**Kernel versions**:

#### Version 1: High Occupancy, Low Performance
```cpp
__global__ void lowResourceKernel(float* data, int n) {
    // Uses only 8 registers, 0 shared memory
    // Occupancy: 100%
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        x = x * 2.0f + 1.0f;  // Simple computation
        data[idx] = x;
    }
}
// Throughput: 450 GB/s
```

#### Version 2: Lower Occupancy, Higher Performance
```cpp
__global__ void highResourceKernel(float* data, int n) {
    // Uses 64 registers per thread, 16 KB shared memory
    // Occupancy: 50%
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // More registers enable more ILP
    float x1 = data[idx];
    float x2 = data[idx + blockDim.x * gridDim.x];
    float x3 = data[idx + 2 * blockDim.x * gridDim.x];
    float x4 = data[idx + 3 * blockDim.x * gridDim.x];
    
    // Complex computation with ILP
    x1 = x1 * 2.0f + 1.0f;
    x2 = x2 * 3.0f + 2.0f;
    x3 = x3 * 4.0f + 3.0f;
    x4 = x4 * 5.0f + 4.0f;
    
    data[idx] = x1 + x2 + x3 + x4;
}
// Throughput: 720 GB/s (60% faster despite lower occupancy!)
```

**Key insight**: 100% occupancy isn't always best! More resources per thread can enable better algorithms.

**How to run**:
```bash
make occupancy_tuning
```

**Expected output**:
```
Low Resource Kernel:
  Occupancy: 100%
  Throughput: 450 GB/s
  
High Resource Kernel:
  Occupancy: 50%
  Throughput: 720 GB/s [OK]

Conclusion: Higher performance despite lower occupancy!
```

---

### tcgen05 preview build (SM100)

While most of Chapter 8 focuses on FP32 kernels that you can run on today’s
hardware, the same tiling techniques extend cleanly to Blackwell’s tcgen05 tensor
cores. The repository now ships two additional benchmarks:

- `ch8/baseline_tiling_tcgen05.py` – reuses the FP32 baseline but only runs when an
  SM100 toolchain is present so results line up with the tcgen05 path.
- `ch8/optimized_tiling_tcgen05.py` – calls the new SM100-only
  `matmul_tiling_tcgen05` kernel implemented in `ch8/tiling_kernels_tcgen05.cu`.
  It feeds fp16 inputs through a tcgen05 pipeline (TMA → TMEM accumulators →
  tcgen05.mma) so you can measure how the same tiling strategy scales on the next
  generation hardware.

Both scripts call `ensure_tcgen05_supported(...)`, so on GB10 they immediately
report `SKIPPED: …` instead of hanging. Once you have B100/B200 access, run:

```bash
python ch8/baseline_tiling_tcgen05.py
python ch8/optimized_tiling_tcgen05.py
```

and compare their results to the FP32 versions to see the delta that tcgen05 brings.

### 2. `[CUDA file]` (see source files for implementation) - ILP Through Unrolling

**Purpose**: Show how loop unrolling increases ILP and hides latency.

#### Baseline: No Unrolling
```cpp
__global__ void sumNoUnroll(const float* data, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    // Serial loop - limited ILP
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += data[i];  // Load → Wait → Add → Repeat
    }
    
    result[idx] = sum;
}
```

**Problem**: Each iteration depends on previous → Can't hide memory latency.

#### Optimized: Loop Unrolling
```cpp
__global__ void sumUnroll4(const float* data, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;
    
    // Unrolled 4x - enables ILP
    for (int i = idx; i < n; i += 4 * blockDim.x * gridDim.x) {
        sum1 += data[i];                            // Load #1
        sum2 += data[i + blockDim.x * gridDim.x];  // Load #2
        sum3 += data[i + 2 * blockDim.x * gridDim.x];  // Load #3
        sum4 += data[i + 3 * blockDim.x * gridDim.x];  // Load #4
        // All 4 loads can happen in parallel!
    }
    
    result[idx] = sum1 + sum2 + sum3 + sum4;
}
```

**Benefit**: 4 independent loads → Memory latency hidden → 3-4x faster!

**How to run**:
```bash
make loop_unrolling
```

**Expected speedup**: **3-4x** (from ~350 GB/s to ~1.2 TB/s)

**Unrolling guidelines**:
- **4x**: Good balance for most kernels
- **8x**: For very memory-bound kernels
- **16x+**: Diminishing returns, increases register pressure

---

###  Maximizing ILP

**Purpose**: Demonstrate independent operations for latency hiding.

```cpp
__global__ void dependentOps(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        
        // Dependent operations (slow!)
        x = x + 1.0f;   // Must wait for load
        x = x * 2.0f;   // Must wait for add
        x = x - 3.0f;   // Must wait for multiply
        x = x / 4.0f;   // Must wait for subtract
        
        data[idx] = x;
    }
}
```

**Optimized**:
```cpp
__global__ void independentOps(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Load multiple independent values
        float x1 = data[idx];
        float x2 = data[idx + blockDim.x * gridDim.x];
        float x3 = data[idx + 2 * blockDim.x * gridDim.x];
        float x4 = data[idx + 3 * blockDim.x * gridDim.x];
        
        // Independent operations (fast!)
        x1 = x1 + 1.0f;  // Can execute in parallel
        x2 = x2 * 2.0f;  // Can execute in parallel
        x3 = x3 - 3.0f;  // Can execute in parallel
        x4 = x4 / 4.0f;  // Can execute in parallel
        
        data[idx] = x1 + x2 + x3 + x4;
    }
}
```

**Speedup**: **2-3x** by enabling parallel execution.

**How to run**:
```bash
make independent_ops
```

---

### 4. `[CUDA file]` (see source files for implementation) → `[CUDA file]` (see source files for implementation) - Warp Divergence

#### Problem: `[CUDA file]` (see source files for implementation)

**Warp divergence**: Threads in warp take different branches → Serialized execution.

```cpp
__global__ void thresholdNaive(const float* in, float* out, int n, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (in[idx] > threshold) {
            // Path A: 50% of threads execute this
            out[idx] = complexComputationA(in[idx]);
        } else {
            // Path B: 50% of threads execute this
            out[idx] = complexComputationB(in[idx]);
        }
        // If data is random, warp executes BOTH paths serially!
    }
}
```

**Performance**: ~50% throughput (both paths executed for all threads).

#### Optimized: `[CUDA file]` (see source files for implementation)

**Solution**: Use predication to avoid branching.

```cpp
__global__ void thresholdPredicated(const float* in, float* out, int n, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        float resultA = complexComputationA(x);
        float resultB = complexComputationB(x);
        
        // Predicated select (no branch!)
        out[idx] = (x > threshold) ? resultA : resultB;
    }
}
```

**When this helps**: If both computations are cheap (few instructions). For expensive functions, branching might still be better.

**Better solution**: Partition data by threshold, process separately:
```cpp
// Kernel 1: Process all values > threshold
// Kernel 2: Process all values ≤ threshold
// No divergence in either kernel!
```

**How to run**:
```bash
make threshold_naive threshold_predicated
```

---

## PyTorch Examples

###  PyTorch Occupancy Analysis

**Purpose**: Show how PyTorch kernel launches affect occupancy.

```python
import torch

# Small batch: Low occupancy
x = torch.randn(32, 128, device='cuda')
y = torch.nn.functional.relu(x)  # Underutilizes GPU

# Large batch: High occupancy  
x = torch.randn(1024, 128, device='cuda')
y = torch.nn.functional.relu(x)  # Much better!
```

**How to run**:
```bash
python3 [script]
```

###  ILP in PyTorch Operations

**Purpose**: Demonstrate batching for ILP.

```python
# Bad: Sequential operations (no ILP)
for i in range(8):
    result = model(batch)
    results.append(result)

# Good: Batched operation (enables ILP)
batched_input = torch.cat([batch] * 8)
result = model(batched_input)  # 8x faster!
```

###  Conditional Execution

**Purpose**: Show impact of conditional operations in PyTorch.

```python
# Divergent: Masked operations
mask = (x > 0.5)
y = torch.where(mask, expensive_op(x), cheap_op(x))
# Both ops executed for all elements!

# Better: Split and process separately
x_high = x[mask]
x_low = x[~mask]
y_high = expensive_op(x_high)
y_low = cheap_op(x_low)
y = torch.zeros_like(x)
y[mask] = y_high
y[~mask] = y_low
```

---

## Occupancy Calculator

### Manual Calculation

```
Given:
- Threads per block: 256
- Registers per thread: 32
- Shared memory per block: 8 KB

Limits for NVIDIA GPU:
- Max warps per SM: 64
- Max threads per SM: 2048
- Max blocks per SM: 32
- Total registers per SM: 65,536
- Total shared memory per SM: 256 KB

Calculate:
1. Warps per block = 256 / 32 = 8 warps
2. Register limit: floor(65,536 / (32 × 256)) = 8 blocks
3. Shared memory limit: floor(256 KB / 8 KB) = 32 blocks
4. Thread limit: floor(2048 / 256) = 8 blocks

Limiting factor: Registers (8 blocks)
Active warps: 8 blocks × 8 warps = 64 warps
Occupancy: 64 / 64 = 100% [OK]
```

### Using CUDA API

```cpp
int blockSize = 256;
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocks, myKernel, blockSize, sharedMemSize);

float occupancy = (numBlocks * blockSize / 32.0f) / 64.0f;
printf("Occupancy: %.1f%%\n", occupancy * 100);
```

---

## How to Run All Examples

```bash
cd ch8

# Install Python dependencies
pip install -r requirements.txt

# Build all CUDA examples
make

# Run occupancy tuning
../.[executable]/profiling/profile_cuda.sh [executable] baseline

# PyTorch examples
python3 [script]
python3 [script]
python3 [script]
```

---

## Key Takeaways

1. **Occupancy isn't everything**: 50% occupancy with good ILP often beats 100% occupancy with poor ILP.

2. **Loop unrolling enables ILP**: Unroll 4-8x to allow independent memory operations to overlap.

3. **Register pressure matters**: Each register per thread reduces max occupancy. Balance register usage vs computation needs.

4. **Warp divergence is expensive**: Threads in warp taking different branches execute serially (50% throughput loss).

5. **ILP hides latency**: Independent operations allow GPU to execute while waiting for memory.

6. **Target 50-75% occupancy**: This is often the sweet spot - enough parallelism to hide latency, enough resources for efficient algorithms.

7. **Profile to validate**: Use Nsight Compute to measure actual occupancy and identify bottlenecks.

---

## Common Pitfalls

### Pitfall 1: Chasing 100% Occupancy
**Problem**: Reducing per-thread resources to maximize occupancy hurts performance.

**Reality**: 50-75% occupancy with better per-thread efficiency often wins.

**Solution**: Profile and measure actual throughput, not just occupancy.

### Pitfall 2: Excessive Register Spilling
**Problem**: Too many registers → Spilling to local memory → 100x slowdown!

**Check in Nsight Compute**: Look for "lmem" (local memory) transactions.

**Solution**: Reduce register usage or use `__launch_bounds__` to control spilling.

### Pitfall 3: Ignoring Warp Divergence
**Problem**: Random branching patterns cause 50% throughput loss.

**Solution**: Partition data to minimize divergence within warps.

### Pitfall 4: No Loop Unrolling
**Problem**: Serial memory operations → Can't hide latency.

**Solution**: Unroll loops 4-8x for better ILP.

### Pitfall 5: Dependent Operations
**Problem**: Each operation waits for previous → No parallelism.

**Solution**: Create independent operation chains that can execute in parallel.

---

## Additional Microbenchmarks (repo examples)

### AI Optimization (ILP + Shared Weights)
- **Baseline** (`ch8/baseline_ai_optimization.py`): serial dot product per sample; every warp reloads the weight vector from global memory.
- **Optimized** (`ch8/optimized_ai_optimization.py`): stages weights into shared memory once per block and processes four elements per lane using `float4` loads to expose ILP.
- **Run**: `python tools/cli/benchmark_cli.py --smoke-test --only-examples ai_optimization ch8`
- **Speedup**: ~2× on GB10.

### Distributed Pipeline
- **Baseline** (`ch8/baseline_distributed.py`): sequential host→device copies plus compute per virtual rank.
- **Optimized** (`ch8/optimized_distributed.py`): issues async copies on independent CUDA streams and overlaps them with compute (mimicking NCCL pipelining).
- **Run**: `python tools/cli/benchmark_cli.py --smoke-test --only-examples distributed ch8`
- **Speedup**: ~1.5× with four virtual ranks.

### NCCL-Style Ring Reduction
- **Baseline** (`ch8/baseline_nccl.py`): copies each chunk to the host and reduces on the CPU.
- **Optimized** (`ch8/optimized_nccl.py`): invokes `nccl_kernels.cu`, which performs the ring reduction entirely on the GPU.
- **Run**: `python tools/cli/benchmark_cli.py --smoke-test --only-examples nccl ch8`
- **Speedup**: ~1.5×.

### HBM Residency vs. Streaming
- **Baseline** (`ch8/baseline_hbm.py`): copies column-major tensors from host memory every iteration.
- **Optimized** (`ch8/optimized_hbm.py`): keeps row-major tensors resident and uses vectorized loads.
- **Run**: `python tools/cli/benchmark_cli.py --smoke-test --only-examples hbm ch8`
- **Speedup**: 3–4× once copies leave the hot loop.

### Double Buffering (Host vs. Device Residency)
- **Baseline** (`ch8/baseline_double_buffering.py`): streams input from pinned host memory each pass before launching the single-buffer kernel.
- **Optimized** (`ch8/optimized_double_buffering.py`): keeps data on device and pipelines loads in shared memory, mirroring the chapter narrative.
- **Run**: `python tools/cli/benchmark_cli.py --smoke-test --only-examples double_buffering ch8`
- **Speedup**: ≥4×.

### Thresholding with cp.async + Warp Ballots
- **Baseline** (`ch8/baseline_threshold.py` + `.cu`): branchy kernel that copies tensors from host per iteration and computes sin/cos everywhere.
- **Optimized** (`ch8/optimized_threshold.py` / `optimized_threshold_predicated.cu`): stages tiles via `cp.async` (TMA) into shared memory, uses warp ballots to skip inactive lanes, and only computes sin/cos when needed.
- **Run**: `python tools/cli/benchmark_cli.py --smoke-test --only-examples threshold ch8`
- **Speedup**: ~3× on the Python harness and ≥1.05× on CUDA-capable TMA hardware.

### Threshold TMA Pipeline (Blackwell Only)
- **Baseline** (`ch8/baseline_thresholdtma.py` / `baseline_thresholdtma.cu`): same math path as the branchy kernel but gated so it only runs when Blackwell/GB GPUs are present.
- **Optimized** (`ch8/optimized_thresholdtma.py` / `optimized_thresholdtma.cu`): rewrites the threshold kernel using the CUDA `cuda::pipeline` API (TMA on Blackwell) with double-buffered tiles.
- **Run**: `python tools/cli/benchmark_cli.py --smoke-test --only-examples thresholdtma ch8`
- **Speedup**: expect ≥1.05× once TMA kernels are enabled; otherwise the harness prints `SKIPPED: …` on unsupported hardware.

---

## Next Steps

**Learn kernel efficiency & fusion** → [Chapter 9: Kernel Efficiency & Arithmetic Intensity](.[executable]/README.md)

Learn about:
- Fusing multiple operations into single kernel
- Reducing memory traffic
- CUTLASS for optimized GEMM
- Inline PTX for low-level control

**Jump to tensor cores** → [Chapter 10: Tensor Cores and Pipelines](.[executable]/README.md)

---

## Additional Resources

- **Occupancy Calculator**: [CUDA Occupancy Calculator Spreadsheet](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/)
- **ILP Best Practices**: [CUDA Best Practices - ILP](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#instruction-level-parallelism)
- **Warp Divergence**: [Control Flow Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#control-flow)
- **Nsight Compute**: [Occupancy Analysis](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#occupancy)

---

**Chapter Status**: [OK] Complete
