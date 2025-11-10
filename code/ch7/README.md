# Chapter 7: Memory Access Patterns

## Overview

Memory bandwidth is often the limiting factor in GPU performance. This chapter teaches you how to access memory efficiently through coalescing, vectorization, shared memory, and proper access patterns. These optimizations typically deliver 2-10x speedups and are essential for any high-performance CUDA code.

## Learning Objectives

After completing this chapter, you can:

- [OK] Understand memory coalescing and achieve near-peak bandwidth
- [OK] Use vectorized loads/stores for 4x memory throughput
- [OK] Leverage shared memory for data reuse
- [OK] Eliminate bank conflicts in shared memory
- [OK] Implement tiled algorithms for locality
- [OK] Measure and optimize memory bandwidth utilization

## Prerequisites

**Previous chapters**:
- [Chapter 2: NVIDIA GPU Hardware](.[executable]/README.md) - memory hierarchy
- [Chapter 6: CUDA Basics](.[executable]/README.md) - thread indexing

**Required**: Understanding of memory hierarchy and cache behavior

## Memory Hierarchy Review

```
Speed (cycles to access):
Registers:        1 cycle
Shared Memory:    ~30 cycles
L1 Cache:         ~30 cycles
L2 Cache:         ~200 cycles
HBM3e (Global):   ~400 cycles  ← Most data lives here
```

**Key insight**: Accessing global memory is 400x slower than registers. Optimization goal: minimize global memory accesses, maximize reuse.

---

## Examples: Memory Access Patterns

### 1. `[CUDA file]` (see source files for implementation) → `[CUDA file]` (see source files for implementation) - Vectorization

#### Baseline: `[CUDA file]` (see source files for implementation)

**Problem**: Scalar loads/stores underutilize memory bus width.

```cpp
__global__ void copyScalar(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];  // 4-byte load + 4-byte store
    }
}
```

**Bandwidth**: ~1.2 TB/s on NVIDIA GPU (15% of peak!)

#### Optimized: `[CUDA file]` (see source files for implementation)

**Solution**: Load 16 bytes (float4) per transaction.

```cpp
__global__ void copyVectorized4(const float4* in, float4* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];  // 16-byte load + 16-byte store
    }
}
```

**Bandwidth**: ~4.8 TB/s on NVIDIA GPU (60% of peak!) [OK]

**Speedup**: **4x faster!**

**NVIDIA GPU optimization**: NVIDIA GPU supports 32-byte vectorized loads (float8):

```cpp
// CUDA 13 / NVIDIA GPU: 32-byte aligned loads
struct alignas(32) Float8 {
    float x0, y0, z0, w0;
    float x1, y1, z1, w1;
};

__global__ void copyVectorized8(const Float8* in, Float8* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];  // 32-byte load!
    }
}
```

**Bandwidth**: ~6.5 TB/s (81% of peak!) [OK][OK]

**How to run**:
```bash
make scalar_copy vectorized_copy
```

---

### 2. `[CUDA file]` (see source files for implementation) → `[CUDA file]` (see source files for implementation) - Memory Coalescing

#### Baseline: `[CUDA file]` (see source files for implementation)

**Problem**: Strided access pattern prevents coalescing.

```cpp
__global__ void copyUncoalesced(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Stride of 32 (warp size) - terrible pattern!
    int access_idx = idx * 32;
    if (access_idx < n) {
        out[access_idx] = in[access_idx];
    }
}
```

**Why it's slow**: Each thread in warp accesses different cache line → 32 transactions instead of 1!

**Bandwidth**: ~150 GB/s (2% of peak!) ERROR: #### Optimized: `[CUDA file]` (see source files for implementation)

**Solution**: Sequential access pattern within warp.

```cpp
__global__ void copyCoalesced(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Threads in warp access consecutive addresses
    if (idx < n) {
        out[idx] = in[idx];
    }
}
```

**Why it's fast**: All threads in warp access same cache line → 1 transaction!

**Bandwidth**: ~3.8 TB/s (48% of peak!) [OK]

**Speedup**: **25x faster** than uncoalesced!

**How to run**:
```bash
make uncoalesced_copy coalesced_copy
```

**Key rule**: Threads in a warp should access consecutive memory addresses.

---

### 3. `[CUDA file]` (see source files for implementation) → `[CUDA file]` (see source files for implementation) - Bank Conflicts

#### Baseline: `[CUDA file]` (see source files for implementation)

**Problem**: Shared memory bank conflicts during transpose.

```cpp
__global__ void transposeNaive(const float* in, float* out, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM];  // 32×32 tile
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load into shared memory (coalesced)
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    __syncthreads();
    
    // Write transposed (bank conflicts!)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    out[y * height + x] = tile[threadIdx.x][threadIdx.y];  // Conflict!
}
```

**Problem**: Reading `tile[threadIdx.x][threadIdx.y]` causes bank conflicts:
- Thread 0 reads tile[0][0] → Bank 0
- Thread 1 reads tile[1][0] → Bank 1
- ...
- Thread 31 reads tile[31][0] → Bank 31
- But then thread 0 reads tile[0][1] → Bank 0 again (conflict!)

**Bandwidth**: ~2.1 TB/s (limited by bank conflicts)

#### Optimized: `[CUDA file]` (see source files for implementation)

**Solution**: Add padding to avoid bank conflicts.

```cpp
__global__ void transposePadded(const float* in, float* out, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 padding!
    
    // Same load
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    __syncthreads();
    
    // Now reads are conflict-free
    out[y * height + x] = tile[threadIdx.x][threadIdx.y];
}
```

**Why it works**: Padding shifts columns to different banks.

**Bandwidth**: ~4.2 TB/s (53% of peak!) [OK]

**Speedup**: **2x faster** than naive!

**How to run**:
```bash
make transpose_naive transpose_padded
```

---

### 4. `[CUDA file]` (see source files for implementation) → `[CUDA file]` (see source files for implementation) - Shared Memory Tiling

#### Baseline: `[CUDA file]` (see source files for implementation)

**Problem**: Each element loaded from global memory multiple times.

```cpp
__global__ void matmulNaive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];  // K global loads per element!
        }
        C[row * N + col] = sum;
    }
}
```

**Problem**: For M=N=K=1024, each element of A and B loaded 1024 times from global memory!

**FLOPS**: ~180 GFLOPS (0.009% of peak!)

#### Optimized: `[CUDA file]` (see source files for implementation)

**Solution**: Load tiles into shared memory, reuse.

```cpp
__global__ void matmulTiled(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    
    // Process tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory
        As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();
        
        // Compute using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}
```

**Benefit**: Each tile loaded once, reused TILE_SIZE times!

**FLOPS**: ~2,100 GFLOPS (12x faster!)

**How to run**:
```bash
make naive_matmul tiled_matmul
[executable] 1024 1024 1024
[executable] 1024 1024 1024
```

**Note**: Chapter 10 covers tensor cores for 100x faster matmul!

---

### 5. `[CUDA file]` (see source files for implementation) → [source file] - Memory Access Patterns

**Purpose**: Demonstrate impact of access patterns on cache efficiency.

**Baseline**: Random lookups (poor cache utilization)
**Optimized**: Sorted lookups (good cache utilization)

**How to run**:
```bash
make naive_lookup optimized_lookup
```

---

## PyTorch Examples

###  Memory Layout

**Purpose**: Show how PyTorch memory layout affects performance.

```python
# Row-major (contiguous)
A = torch.randn(1024, 1024, device='cuda')
B = torch.randn(1024, 1024, device='cuda')
C = torch.matmul(A, B)  # Fast!

# Column-major (transposed, non-contiguous)
A_t = A.t()  # View, not copy
C = torch.matmul(A_t, B)  # Slower! (non-contiguous access)

# Fix: Make contiguous
A_t_cont = A.t().contiguous()
C = torch.matmul(A_t_cont, B)  # Fast again!
```

**How to run**:
```bash
python3 [script]
```

###  Vectorized Operations

**Purpose**: Compare element-wise vs vectorized ops.

```python
# Slow: Element-wise in Python loop
for i in range(n):
    C[i] = A[i] + B[i]  # Terrible!

# Fast: Vectorized operation
C = A + B  # 1000x faster!
```

---

## Memory Bandwidth Analysis

### Measuring Bandwidth

Use common profiling tools:

```bash
../.[executable]/profiling/profile_cuda.sh [executable] baseline
```

**In Nsight Compute**, look for:
- **Memory Throughput**: Actual GB/s achieved
- **Theoretical Bandwidth**: Peak for your GPU (8 TB/s for NVIDIA GPU)
- **Efficiency %**: Actual / Theoretical

### Performance Targets (NVIDIA GPU)

| Access Pattern | Bandwidth | Efficiency | Status |
|----------------|-----------|------------|--------|
| Scalar loads | 1.2 TB/s | 15% | ERROR: Poor |
| Vectorized (float4) | 4.8 TB/s | 60% | [OK] Good |
| Vectorized (float8/NVIDIA GPU) | 6.5 TB/s | 81% | [OK][OK] Excellent |
| Uncoalesced | 0.15 TB/s | 2% | ERROR: Terrible |
| Coalesced | 3.8 TB/s | 48% | [OK] Good |

**Reality check**: 40-60% of peak bandwidth is excellent for real kernels!

### Roofline Model

**Roofline** = min(Peak_Compute, Peak_Bandwidth × Arithmetic_Intensity)

For memory-bound kernels:
```
Achieved_FLOPS = Achieved_Bandwidth × (FLOPS / Byte)
```

**Example**: Vector addition
- Arithmetic intensity: 1 FLOP / 12 bytes (load A, load B, store C)
- Peak bandwidth: 8 TB/s
- Roofline: 8000 GB/s × (1/12) = 667 GFLOPS maximum
- Actual: ~600 GFLOPS [OK] (near roofline!)

---

## How to Run All Examples

```bash
cd ch7

# Build all examples
make

# Run memory access pattern comparisons



[executable] 1024 1024 1024    # No tiling
[executable] 1024 1024 1024    # 12x faster

# Profile to see bandwidth
../.[executable]/profiling/profile_cuda.sh [executable] baseline

# PyTorch examples
python3 [script]
python3 [script]
```

---

## Key Takeaways

1. **Memory is the bottleneck**: Most GPU kernels are memory-bound, not compute-bound. Optimize memory first!

2. **Coalescing is critical**: Ensure threads in a warp access consecutive addresses. 25x speedup from this alone!

3. **Vectorization gives free 4x**: Use float4 (or float8 on NVIDIA GPU) for 4-8x memory throughput.

4. **Shared memory enables reuse**: For algorithms with temporal locality (matmul, convolution), tiling with shared memory is essential.

5. **Bank conflicts matter**: Add padding (+1) to shared memory arrays to avoid conflicts during transpose operations.

6. **40-60% efficiency is good**: Don't expect to hit 100% of theoretical bandwidth. 40-60% is excellent for complex kernels.

7. **Profile, don't guess**: Use Nsight Compute to measure actual bandwidth and identify bottlenecks.

---

## Common Pitfalls

### Pitfall 1: Strided Access Patterns
**Problem**: Accessing every Nth element breaks coalescing.

**Bad**:
```cpp
data[idx * stride]  // Uncoalesced if stride > 1
```

**Good**:
```cpp
data[idx]  // Coalesced
```

### Pitfall 2: Structure of Arrays (SoA) vs Array of Structures (AoS)

**AoS (bad for GPU)**:
```cpp
struct Point { float x, y, z; };
Point points[N];
// Thread i accesses points[i].x → stride of 12 bytes!
```

**SoA (good for GPU)**:
```cpp
float x[N], y[N], z[N];
// Thread i accesses x[i] → consecutive!
```

### Pitfall 3: Forgetting Alignment
**Problem**: Unaligned loads are slower.

**Solution**: Ensure data is aligned to 16 or 32 bytes:
```cpp
float* data;
cudaMalloc(&data, size);  // Automatically aligned
// Or explicit:
posix_memalign((void**)&data, 32, size);
```

### Pitfall 4: Shared Memory Overuse
**Problem**: Using more shared memory reduces occupancy.

**Check**: NVIDIA GPU has 256 KB shared memory per SM. If kernel uses >16 KB shared per block, max occupancy drops below 100%.

**Solution**: Use less shared memory or increase block size.

### Pitfall 5: Non-Contiguous PyTorch Tensors
**Problem**: Transposed or sliced tensors are non-contiguous.

**Check**:
```python
if not tensor.is_contiguous():
    tensor = tensor.contiguous()
```

---

## Next Steps

**Master occupancy and ILP** → [Chapter 8: Occupancy and ILP](.[executable]/README.md)

Learn about:
- Occupancy tuning for latency hiding
- Instruction-level parallelism (ILP)
- Register pressure management
- Warp divergence mitigation

**Jump to kernel efficiency** → [Chapter 9: Kernel Efficiency & Arithmetic Intensity](.[executable]/README.md)

---

## Additional Resources

- **Memory Coalescing**: [CUDA Best Practices Guide - Coalescing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
- **Shared Memory**: [CUDA Programming Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- **Bank Conflicts**: [Shared Memory Bank Conflicts](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- **Roofline Model**: [Roofline Performance Model](https://en.wikipedia.org/wiki/Roofline_model)

---

**Chapter Status**: [OK] Complete
