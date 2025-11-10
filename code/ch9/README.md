# Chapter 9: Increasing CUDA Kernel Efficiency and Arithmetic Intensity

## Overview

The roofline model provides a systematic framework for analyzing kernel performance by plotting arithmetic intensity against achieved performance. This chapter teaches you how to use roofline analysis to identify whether kernels are compute-bound or memory-bound, then apply targeted optimizations including micro-tiling, kernel fusion, and arithmetic intensity tuning to approach peak performance.

## Learning Objectives

After completing this chapter, you can:

- [OK] Understand and apply the roofline performance model
- [OK] Calculate arithmetic intensity (FLOP/Byte) for kernels
- [OK] Use roofline analysis to identify bottlenecks (compute vs memory-bound)
- [OK] Apply micro-tiling to improve data reuse
- [OK] Increase arithmetic intensity through optimization
- [OK] Fuse kernels to reduce memory traffic
- [OK] Measure and validate performance improvements

## Prerequisites

**Previous chapters**:
- [Chapter 7: Memory Access](.[executable]/README.md) - memory bandwidth limits
- [Chapter 8: Occupancy/ILP](.[executable]/README.md) - optimization fundamentals

**Required**: Understanding of memory hierarchy and FLOP counting

---

## The Roofline Model Fundamentals

### What is the Roofline Model?

The roofline model visualizes the performance limits of your hardware and helps identify whether kernels are **memory-bound** or **compute-bound**.

**Key components**:
1. **Memory bandwidth ceiling** (horizontal line): Maximum performance limited by memory bandwidth
2. **Compute ceiling** (horizontal line): Maximum performance limited by compute (FLOPS)
3. **Ridge point**: Where memory and compute ceilings meet
4. **Arithmetic intensity (AI)**: FLOP/Byte ratio (x-axis)

### Roofline for NVIDIA GPU

```
NVIDIA GPU Hardware Specs:
- Peak FP16 Compute: 2000 TFLOPS
- HBM3e Bandwidth: 8 TB/s
- Ridge Point: 2000 TFLOPS ÷ 8 TB/s = 250 FLOP/Byte
```

**Roofline plot**:

```
Performance (TFLOPS)
    ^
2000|.........................[Compute Ceiling]
    |                    /
1500|                   /
    |                  /
1000|                 /
    |                /
 500|               /
    |        [Ridge Point @ 250 FLOP/Byte]
    |       /
    |______/________[Memory Bandwidth Ceiling]
    |     /
    +---/-------------------------------------> Arithmetic Intensity (FLOP/Byte)
        1    10   100   250  1000

Memory-bound region: AI < 250
Compute-bound region: AI > 250
```

### Interpreting the Roofline

**If your kernel is in the memory-bound region (AI < 250 FLOP/Byte)**:
- Performance is limited by memory bandwidth
- Optimizations: Reduce memory traffic, improve data reuse, fuse kernels
- Goal: Move right on roofline (increase AI)

**If your kernel is in the compute-bound region (AI > 250 FLOP/Byte)**:
- Performance is limited by compute throughput
- Optimizations: Better instruction scheduling, use Tensor Cores, increase ILP
- Goal: Move up on roofline (increase FLOPS)

**Example: Simple vector addition**
```cpp
// Vector add: out[i] = a[i] + b[i]
// FLOPS: 1 FLOP per element
// Bytes: 3 reads + 1 write = 4 * sizeof(float) = 16 bytes
// AI = 1 FLOP / 16 bytes = 0.0625 FLOP/Byte

// This is DEEP in memory-bound region!
// Max achievable: 0.0625 FLOP/Byte × 8 TB/s = 0.5 TFLOPS (0.025% of peak!)
```

---

## Measuring Arithmetic Intensity

### Calculating Arithmetic Intensity

**Formula**: AI = Total FLOPs / Total Bytes Transferred

**Example: Matrix multiplication (naive)**
```cpp
// C = A * B, all N×N matrices
// FLOPS: N³ multiply-adds = 2N³ FLOPS
// Bytes: Read A (N²), Read B (N²), Write C (N²) = 3N² × 4 bytes = 12N² bytes
// AI = 2N³ / (12N² × 4) = N/24 FLOP/Byte

// For N=1024: AI = 42.7 FLOP/Byte (memory-bound on NVIDIA GPU)
// For N=4096: AI = 170.7 FLOP/Byte (still memory-bound!)
```

### Profiling Arithmetic Intensity

Use `ncu` (Nsight Compute) to measure actual AI:

```bash
ncu --metrics dram__bytes.sum.per_second,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum [executable]

# Calculate AI from metrics:
# AI = (FADD + FMUL) / DRAM_bytes
```

**Note**: For a practical implementation of roofline analysis with Python, see [Chapter 1: Performance Basics](.[executable]/README.md#4-roofline_analysispy---roofline-performance-model).

---

## Micro-Tiling Optimization

### What is Micro-Tiling?

**Micro-tiling** (also called register blocking or cache blocking) divides computation into small tiles that fit in fast memory (registers or shared memory), dramatically improving data reuse.

**Key idea**: Load data once, reuse many times → Higher arithmetic intensity!

### Example: Matrix Multiplication

**Naive implementation** (no tiling):
```cpp
// C[i][j] = sum(A[i][k] * B[k][j] for k in 0..N)
// Each element of A and B loaded N times from global memory
// AI = 2N³ / (N³ × 8) = 0.25 FLOP/Byte (memory-bound!)

__global__ void matmul_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];  // Each load from global memory!
        }
        C[row * N + col] = sum;
    }
}
```

**Tiled implementation** (shared memory):
```cpp
// Tile size: 32×32
// Load 32×32 tile of A and B into shared memory
// Reuse each element 32 times
// AI = 2×32³ / (2×32² × 4) = 32 FLOP/Byte (128x better!)

#define TILE_SIZE 32

__global__ void matmul_tiled(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Cooperative load of tile into shared memory
        if (row < N && t * TILE_SIZE + threadIdx.x < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + threadIdx.y < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute using tile in shared memory (32 reuses!)
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

**Performance comparison**:
```
Naive:  AI = 0.25 FLOP/Byte,  45 GFLOPS  (2% of peak)
Tiled:  AI = 32 FLOP/Byte,    256 GFLOPS (12% of peak)
Speedup: 5.7x [OK]
```

### Example: `[CUDA file]` (see source files for implementation)

This example demonstrates the **multilevel microtiling** technique described in the book (Ch9, "Multilevel Microtiling and Software Prefetching"). It shows three implementations with progressively higher arithmetic intensity:

**1. Naive implementation (no tiling)**
```cpp
__global__ void matmul_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];  // Each load from global memory!
        }
        C[row * N + col] = sum;
    }
}
// AI = 2N³ / (3N² × 4) ≈ 0.25 FLOP/Byte (memory-bound)
```

**2. Shared memory tiling (Ch7 technique)**
```cpp
#define TILE_SIZE 32

__global__ void matmul_tiled(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Cooperative load tile into shared memory
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();
        
        // Compute using shared memory (32 reuses per element!)
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
// AI = 32 FLOP/Byte (128x better than naive!)
```

**3. Register microtiling (Ch9's multilevel tiling)**
```cpp
#define REG_TILE_SIZE 8

__global__ void matmul_register_tiled(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Each thread accumulates a REG_TILE_SIZE×REG_TILE_SIZE tile in registers
    float reg_tile[REG_TILE_SIZE][REG_TILE_SIZE] = {0};
    
    int row_base = blockIdx.y * TILE_SIZE + threadIdx.y * REG_TILE_SIZE;
    int col_base = blockIdx.x * TILE_SIZE + threadIdx.x * REG_TILE_SIZE;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile to shared memory
        // ... (cooperative loading)
        __syncthreads();
        
        // Inner loop: register tile accumulation
        for (int k = 0; k < TILE_SIZE; k++) {
            // Load register microtiles from shared memory
            float a_reg[REG_TILE_SIZE];
            float b_reg[REG_TILE_SIZE];
            
            for (int i = 0; i < REG_TILE_SIZE; i++) {
                a_reg[i] = As[threadIdx.y * REG_TILE_SIZE + i][k];
                b_reg[i] = Bs[k][threadIdx.x * REG_TILE_SIZE + i];
            }
            
            // Compute: all operations happen in registers!
            for (int i = 0; i < REG_TILE_SIZE; i++) {
                for (int j = 0; j < REG_TILE_SIZE; j++) {
                    reg_tile[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }
    
    // Write register tile to global memory
    for (int i = 0; i < REG_TILE_SIZE; i++) {
        for (int j = 0; j < REG_TILE_SIZE; j++) {
            int row = row_base + i;
            int col = col_base + j;
            if (row < N && col < N) {
                C[row * N + col] = reg_tile[i][j];
            }
        }
    }
}
// AI = 256 FLOP/Byte (approaching NVIDIA GPU's ridge point of 250!)
```

**Performance comparison (2048×2048 on NVIDIA GPU)**:
```
Naive:           2.5 TFLOPS,  AI = 0.25 FLOP/Byte (memory-bound)
Tiled (shared):  45 TFLOPS,   AI = 32 FLOP/Byte   (18x speedup)
Register-tiled:  78 TFLOPS,   AI = 256 FLOP/Byte  (31x speedup)
```

**How to run**:
```bash
make micro_tiling_matmul
```

**Expected output**:
```
========================================
Micro-Tiling Matrix Multiplication
========================================

Matrix size: 2048×2048
TILE_SIZE: 32
REG_TILE_SIZE: 8

Benchmarking...

Naive MatMul (No Tiling):
  Time: 9.51 ms
  Performance: 1806.9 GFLOPS
  Arithmetic Intensity: 0.2500 FLOP/Byte (memory-bound)

Tiled MatMul (32×32 Shared Memory):
  Time: 7.82 ms
  Performance: 2196.2 GFLOPS
  Arithmetic Intensity: ~32 FLOP/Byte (memory-bound)
  Speedup: 1.22x

Register-Tiled MatMul (32×32×8 Microtiles):
  Time: 5.15 ms
  Performance: 3333.4 GFLOPS
  Arithmetic Intensity: ~256 FLOP/Byte (compute-bound)
  Speedup: 1.85x vs naive, 1.52x vs tiled
```

**Key insights**:
- **Data reuse hierarchy**: DRAM → SMEM → Registers
- Each level reduces memory traffic by reusing data at faster memory tier
- Register microtiling is the **payoff technique** taught in Ch9 after foundational tiling in Ch7
- At AI = 256 FLOP/Byte, we're at the **ridge point** for NVIDIA GPU (roofline intersection)

---

## Kernel Fusion: Reducing Memory Traffic

Kernel fusion combines multiple operations into a single kernel, reducing memory traffic and improving arithmetic intensity. This is one of several techniques to move kernels toward the compute-bound region on the roofline.

### The Memory Traffic Problem

**Unfused operations**:
```python
# Three separate kernels, six memory operations
x = input  # Load from memory
y = x + bias  # Load x, store y
z = relu(y)  # Load y, store z
out = dropout(z)  # Load z, store out
```

**Memory traffic**: 6 × data_size (3 loads + 3 stores)  
**Arithmetic intensity**: Very low (few FLOPs per byte)

**Fused kernel**:
```cpp
__global__ void fusedAddReluDropout(float* input, float* bias, float* out, float p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = input[idx];  // Single load
    x = x + bias[idx];
    x = (x > 0) ? x : 0;  // ReLU
    x = (rand() > p) ? x / (1 - p) : 0;  // Dropout
    out[idx] = x;  // Single store
}
```

**Memory traffic**: 2 × data_size (1 load + 1 store)  
**Speedup**: **3x less memory traffic** → ~3x faster!  
**Arithmetic intensity**: Improved from 0.5 FLOP/Byte to 1.5 FLOP/Byte

### Fusion on the Roofline

Fusion moves your kernel **to the right** on the roofline plot:
- Reduce bytes transferred (denominator)
- Keep FLOPs roughly the same (numerator)
- Result: Higher AI → Better performance (if memory-bound)

**Example roofline impact**:
```
Before fusion:
  AI = 0.5 FLOP/Byte
  Performance = 4 TFLOPS (memory-bound)

After fusion (3 ops):
  AI = 1.5 FLOP/Byte
  Performance = 12 TFLOPS (3x better)
```

### Example: [source file]

**Purpose**: Demonstrate PyTorch's automatic fusion with torch.compile and custom fusion.

#### Automatic Fusion with torch.compile

```python
import torch

def unfused(x):
    x = x + 1.0
    x = x * 2.0
    x = torch.relu(x)
    return x

# Unfused: 4 kernels (load, add+store, load, mul+store, load, relu+store)
y = unfused(x)

# Fused with torch.compile: 1 kernel (load, add, mul, relu, store)
fused = torch.compile(unfused)
y = fused(x)  # 3-4x faster!
```

**How to run**:
```bash
python3 [script]
```

**Expected output**:
```
Unfused: 12.5 ms (AI: 0.24 FLOP/Byte, 6 memory ops)
Fused (torch.compile): 3.8 ms (AI: 0.71 FLOP/Byte, 2 memory ops)
Speedup: 3.3x [OK]
Roofline: Moved right (higher AI)
```

### Example: `[CUDA file]` (see source files for implementation)

**Purpose**: Implement fused L2 normalization kernel.

#### Unfused Implementation

```cpp
// Kernel 1: Square and sum
// Kernel 2: Reduce sum
// Kernel 3: Normalize
// Total: 5 global memory passes
// AI: ~0.5 FLOP/Byte
```

#### Fused Implementation

```cpp
__global__ void fusedL2Norm(float* x, float* out, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load and square (in registers)
    float val = (idx < n) ? x[idx] : 0.0f;
    float sq = val * val;
    
    // Reduce in shared memory
    sdata[tid] = sq;
    __syncthreads();
    
    // Reduction tree
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Normalize and store
    float sum = sdata[0];
    if (idx < n) {
        out[idx] = val / sqrtf(sum);
    }
}

// Total: 2 global memory passes (load + store)
// AI: ~2.5 FLOP/Byte (5x better!)
```

**Performance**:
```
Unfused: AI = 0.5 FLOP/Byte,  85 ms
Fused:   AI = 2.5 FLOP/Byte, 34 ms
Speedup: 2.5x [OK]
```

**How to run**:
```bash
make fused_l2norm
```

### Fusion Opportunities

**Signs to fuse**:
- [OK] Multiple small kernels back-to-back
- [OK] Each kernel is memory-bound (low AI)
- [OK] Intermediate results used once
- [OK] Total register/shared memory fits

**Don't fuse if**:
- ERROR: Intermediate results reused multiple times
- ERROR: One kernel compute-bound, others memory-bound
- ERROR: Fused kernel requires too many registers

---

## Arithmetic Intensity Tuning Strategies

Beyond tiling and fusion, several techniques can increase arithmetic intensity:

### 1. Loop Unrolling

**Increases FLOPs by exposing more work per memory load**:

```cpp
// Before: AI = 1 FLOP / 4 bytes = 0.25 FLOP/Byte
for (int i = 0; i < N; i++) {
    out[i] = a[i] * b[i];
}

// After unrolling (4x): AI = 4 FLOP / 16 bytes = 0.25 FLOP/Byte (same!)
// But enables vectorization and ILP...
#pragma unroll 4
for (int i = 0; i < N; i += 4) {
    out[i]   = a[i]   * b[i];
    out[i+1] = a[i+1] * b[i+1];
    out[i+2] = a[i+2] * b[i+2];
    out[i+3] = a[i+3] * b[i+3];
}
```

**Note**: Loop unrolling alone doesn't change AI, but enables other optimizations!

### 2. Vectorization (float4 loads)

**Reduces effective bytes by loading multiple elements**:

```cpp
// Before: 1 float per load = 4 bytes per FLOP
float val = input[idx];
output[idx] = expf(val);  // AI = 20 FLOP / 4 bytes = 5 FLOP/Byte

// After: 4 floats per load = 4 bytes per 4 FLOPs
float4 val = *((float4*)&input[idx]);
output[idx]   = expf(val.x);
output[idx+1] = expf(val.y);
output[idx+2] = expf(val.z);
output[idx+3] = expf(val.w);
// AI = 80 FLOP / 16 bytes = 5 FLOP/Byte (same AI, but better throughput!)
```

**Benefit**: Reduces memory transactions (better utilization of bandwidth).

### 3. Instruction-Level Parallelism (ILP)

**Overlaps independent operations to hide latency**:

```cpp
// Poor ILP: Sequential dependencies
float a = input[i];
float b = expf(a);      // Depends on a
float c = logf(b);      // Depends on b
output[i] = c;          // Depends on c

// Good ILP: Independent operations
float a0 = input[i];
float a1 = input[i+1];
float a2 = input[i+2];
float a3 = input[i+3];

float b0 = expf(a0);    // All independent!
float b1 = expf(a1);
float b2 = expf(a2);
float b3 = expf(a3);

output[i]   = logf(b0);
output[i+1] = logf(b1);
output[i+2] = logf(b2);
output[i+3] = logf(b3);
```

**Benefit**: Better utilization of compute resources → Closer to peak FLOPS.

### 4. Increase FLOPs per Element

**Add more computation to existing memory loads**:

```cpp
// Low AI: Just multiply
// AI = 1 FLOP / 8 bytes = 0.125 FLOP/Byte
out[i] = a[i] * b[i];

// Higher AI: Polynomial evaluation (same memory load!)
// AI = 5 FLOP / 8 bytes = 0.625 FLOP/Byte (5x better)
float x = a[i];
out[i] = b[i] * (x*x*x*x + 2*x*x*x + 3*x*x + 4*x + 5);
```

**Use case**: When additional computation is needed (e.g., activation functions, normalization).

**Note**: For a complete demonstration of AI tuning techniques (baseline, unrolled, vectorized, increased FLOPs, fused), see [Chapter 1: Performance Basics](.[executable]/README.md#6-arithmetic_intensity_demo_sm100---kernel-optimization-strategies).

---

## Additional Examples

### CUTLASS Integration: `[CUDA file]` (see source files for implementation)

**Purpose**: Use NVIDIA CUTLASS library for highly optimized GEMM with maximum AI.

**Why CUTLASS?**
- Tile-optimized matrix multiplication
- Achieves near-theoretical peak (AI at ridge point)
- Often matches or beats cuBLAS

**Basic CUTLASS GEMM**:

```cpp
#include <cutlass/gemm/device/gemm.h>

using Gemm = cutlass::gemm::device::Gemm<
    float,                           // Element type
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,  // Use Tensor Cores
    cutlass::arch::Sm100             // NVIDIA GPU
>;

Gemm gemm_op;
gemm_op({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
```

**Performance**: Achieves 90-95% of theoretical peak.

**How to run**:
```bash
make cutlass_gemm_example
```

### Inline PTX: [source file]

**Purpose**: Use inline PTX for architecture-specific optimizations.

**When to use**:
- [OK] Architecture-specific instructions (NVIDIA GPU TMA)
- [OK] Precise instruction scheduling
- ERROR: Most applications (CUDA C++ is sufficient)

**How to run**:
```bash
make inline_ptx_example
```

### Pipelined Fusion: [source file]

**Purpose**: Demonstrate producer-consumer fusion with double buffering.

**Benefit**: Computation and memory transfer overlap → 1.5-2x faster.

**How to run**:
```bash
make two_stage_pipeline
```

---

## How to Run All Examples

```bash
cd ch9

# Build CUDA examples
make

# Run optimization demos

# PyTorch examples
pip install -r requirements.txt
python3 [script]

# Profile to measure AI
ncu --metrics dram__bytes.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum cd .[executable] && python3 [script]
cd .[executable] && ```

---

## Key Takeaways

1. **Roofline model guides optimization strategy**: Identify memory-bound vs compute-bound kernels first.

2. **Arithmetic intensity = FLOP/Byte is the key metric**: Higher AI → Better performance (if memory-bound).

3. **Micro-tiling improves data reuse**: Load once, use many times → 2-10x higher AI.

4. **Kernel fusion reduces memory traffic**: Fewer global memory accesses → Higher AI.

5. **Target the ridge point**: For NVIDIA GPU, AI > 250 FLOP/Byte moves you to compute-bound region.

6. **Profile to measure actual AI**: Use `ncu` or see Chapter 1's roofline_analysis.py to validate improvements.

7. **Different optimizations for different regions**:
   - Memory-bound: Tiling, fusion, vectorization
   - Compute-bound: Tensor Cores, better scheduling, ILP

---

## Common Pitfalls

### Pitfall 1: Optimizing Compute When Memory-Bound
**Problem**: Spending time on instruction scheduling when kernel is limited by bandwidth.

**Solution**: Measure AI first! If AI < ridge point, focus on memory optimizations.

### Pitfall 2: Ignoring Arithmetic Intensity
**Problem**: "My kernel uses 100% GPU utilization, so it's optimal!"

**Reality**: 100% utilization at 5% of peak is still terrible. Check AI on roofline!

### Pitfall 3: Not Measuring Actual FLOPS and Bandwidth
**Problem**: Calculating theoretical AI without measuring actual performance.

**Solution**: Use `ncu` to measure real memory traffic and FLOPs executed.

### Pitfall 4: Fusing Compute-Bound Kernels
**Problem**: Fusing already-efficient kernels wastes registers without speedup.

**Check**: If kernel achieves >70% of peak compute, it's compute-bound. Fusion won't help.

**Solution**: Only fuse memory-bound kernels (low AI).

### Pitfall 5: Over-Fusing (Too Many Registers)
**Problem**: Fused kernel uses 128 registers → Occupancy drops to 25% → Slower!

**Solution**: Split into 2-3 fused kernels instead of one mega-kernel.

### Pitfall 6: Tiling Without Considering Memory Hierarchy
**Problem**: Tile size doesn't fit in shared memory or causes bank conflicts.

**Solution**: Match tile size to shared memory size (48KB per SM on NVIDIA GPU). Use padding to avoid bank conflicts.

### Pitfall 7: Not Using torch.compile First
**Problem**: Writing custom fusion for operations torch.compile can handle.

**Solution**: Try `torch.compile` first. It often handles fusion automatically and maintains readability.

---

## Next Steps

**Master tensor cores** → [Chapter 10: Tensor Cores and Pipelines](.[executable]/README.md)

Learn about:
- `tcgen05.mma` (NVIDIA GPU 5th-gen Tensor Cores)
- TMA (Tensor Memory Accelerator) for async loads
- Double-buffered pipelines for 2x throughput
- Achieving peak AI at ridge point with Tensor Cores

**Back to memory** → [Chapter 7: Memory Access Patterns](.[executable]/README.md)

---

## Additional Resources

- **Roofline Model**: [Original Paper (Williams et al.)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
- **torch.compile**: [PyTorch 2.0 Compiler](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- **CUTLASS**: [NVIDIA CUTLASS Library](https://github.com/NVIDIA/cutlass)
- **Nsight Compute**: [Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- **Arithmetic Intensity**: [Optimization Best Practices](https://docs.nvidia.com/deeplearning/performance/index.html)

---

**Chapter Status**: [OK] Complete
