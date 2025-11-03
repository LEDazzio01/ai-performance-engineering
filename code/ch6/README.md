# Chapter 6: CUDA Basics - Your First Kernels

## Overview

This chapter introduces CUDA programming from the ground up. You'll write your first CUDA kernels, understand the thread hierarchy, and learn fundamental parallelization patterns that form the basis for all GPU programming.

## Learning Objectives

After completing this chapter, you can:

- ✅ Write and launch basic CUDA kernels
- ✅ Understand the CUDA thread hierarchy (threads, blocks, grids)
- ✅ Calculate grid and block dimensions for arbitrary problem sizes
- ✅ Use thread indexing to map computations to data
- ✅ Apply basic parallelization patterns
- ✅ Understand occupancy and resource limits

## Prerequisites

**Previous chapters**: 
- [Chapter 1: Performance Basics](../ch1/README.md) - profiling fundamentals
- [Chapter 2: B200 Hardware](../ch2/README.md) - GPU architecture basics

**Required**: Basic C/C++ knowledge, CUDA-capable GPU

## CUDA Thread Hierarchy

Before diving into examples, understand the execution model:

```
Grid (entire kernel launch)
├── Block 0
│   ├── Warp 0 (threads 0-31)
│   ├── Warp 1 (threads 32-63)
│   └── ...
├── Block 1
│   ├── Warp 0
│   └── ...
└── ...

Key constraints (B200):
- Max threads per block: 1024
- Warp size: 32 (threads execute in lock-step)
- Max blocks per SM: 32
- Max warps per SM: 64
```

**Critical concept**: Threads within a warp execute simultaneously (SIMT - Single Instruction, Multiple Threads).

---

## Examples

### 1. `my_first_kernel.cu` - Hello World

**Purpose**: Simplest possible CUDA kernel - print from GPU.

**Code**:
```cpp
__global__ void hello() {
    printf("Hello from thread %d in block %d\n", 
           threadIdx.x, blockIdx.x);
}

int main() {
    hello<<<2, 4>>>();  // 2 blocks, 4 threads per block
    cudaDeviceSynchronize();
    return 0;
}
```

**Key concepts**:
- `__global__`: Kernel function (runs on GPU, called from CPU)
- `<<<blocks, threads>>>`: Launch configuration
- `threadIdx.x`: Thread index within block
- `blockIdx.x`: Block index within grid

**How to run**:
```bash
make my_first_kernel
./my_first_kernel_sm100
```

**Expected output**:
```
Hello from thread 0 in block 0
Hello from thread 1 in block 0
...
Hello from thread 3 in block 1
```

---

### 2. `simple_kernel.cu` - Element-wise Operation

**Purpose**: Demonstrates complete CUDA workflow with memory management and kernel launch configuration from the book.

**What it demonstrates**:
- Pinned memory allocation (`cudaMallocHost`)
- Device memory allocation and H2D/D2H transfers
- Thread indexing calculation
- Grid/block dimension calculation
- In-place element-wise operation

**Code walkthrough**:

```cpp
__global__ void myKernel(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        input[idx] *= 2.0f;  // Scale each element by 2
    }
}

int main() {
    const int N = 1'000'000;
    
    // Allocate pinned host memory (faster H2D/D2H transfers)
    float* h_input = nullptr;
    cudaMallocHost(&h_input, N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;
    }
    
    // Allocate device memory
    float* d_input = nullptr;
    cudaMalloc(&d_input, N * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Calculate grid dimensions
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, N);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_input, d_input, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_input);
    cudaFreeHost(h_input);
    return 0;
}
```

**Key concepts**:
- **Grid calculation**: `(N + threadsPerBlock - 1) / threadsPerBlock` ensures full coverage
- **Pinned memory**: `cudaMallocHost` enables faster transfers than `malloc`
- **Synchronization**: `cudaDeviceSynchronize()` waits for kernel completion
- **Bounds checking**: `if (idx < N)` handles cases where grid size > problem size

**How to run**:
```bash
make simple_kernel
./simple_kernel_sm100
```

**Expected output**:
```
✓ Simple kernel succeeded: 1000000 elements scaled by 2.0f
  Configuration: 3907 blocks × 256 threads = 1000192 total threads
```

**Performance characteristics**:
- 1M elements @ 256 threads/block = 3,907 blocks
- Each block processes ~256 elements
- Executes in < 1ms on B200 (memory-bandwidth bound)

---

### 3. `add_sequential.cu` → `add_parallel.cu` - Baseline vs Optimized

**Purpose**: Demonstrate the fundamental parallelization pattern: sequential → parallel.

#### Baseline: `add_sequential.cu`

**Problem**: Single thread does all work (underutilizes GPU).

```cpp
__global__ void addSequential(const float* A, const float* B, float* C, int n) {
    // Only thread 0 of block 0 works!
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < n; i++) {
            C[i] = A[i] + B[i];
        }
    }
}

// Launch: addSequential<<<1, 1>>>(...)
```

**Performance**: ~2,000x slower than parallel version!

#### Optimized: `add_parallel.cu`

**Solution**: Each thread processes one element.

```cpp
__global__ void addParallel(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Launch: 
int threads = 256;
int blocks = (n + threads - 1) / threads;
addParallel<<<blocks, threads>>>(A, B, C, n);
```

**Key pattern**: `idx = blockIdx.x * blockDim.x + threadIdx.x`  
This is the most common indexing pattern in CUDA!

**How to run**:
```bash
make add_sequential add_parallel
./add_sequential_sm100  # Slow
./add_parallel_sm100    # Fast!
```

**Expected speedup**: **~2000x** (1,000,000 elements)

**Why such dramatic speedup?**
- Sequential: 1 thread × 1,000,000 iterations = 1,000,000 operations serially
- Parallel: 3,907 threads × 256 operations each = runs in parallel!

---

### 4. `2d_kernel.cu` - 2D Thread Indexing

**Purpose**: Extend to 2D problems (images, matrices).

**Pattern for 2D grids**:
```cpp
__global__ void process2D(float* matrix, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int idx = row * width + col;  // Row-major indexing
        matrix[idx] = /* computation */;
    }
}

// Launch:
dim3 threads(16, 16);  // 16×16 = 256 threads per block
dim3 blocks((width + 15) / 16, (height + 15) / 16);
process2D<<<blocks, threads>>>(matrix, width, height);
```

**Use cases**: Image processing, matrix operations, convolutions.

**How to run**:
```bash
make 2d_kernel
./2d_kernel_sm100
```

---

### 5. `occupancy_api.cu` - Occupancy Calculation

**Purpose**: Understand occupancy and resource limits.

**What is occupancy?**
```
Occupancy = Active_Warps / Max_Warps_Per_SM
```

Higher occupancy → Better latency hiding → Higher throughput (usually).

**Code**:
```cpp
int blockSize = 256;
int minGridSize, gridSize;

// Calculate optimal block size
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel);

// Calculate occupancy for chosen block size
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, myKernel, blockSize, 0);

float occupancy = (numBlocks * blockSize / 32.0f) / 64.0f;  // 64 warps/SM on B200
printf("Occupancy: %.1f%%\n", occupancy * 100);
```

**How to run**:
```bash
make occupancy_api
./occupancy_api_sm100
```

**Expected output**:
```
Block size: 256
Occupancy: 100% (full SM utilization) ✅
```

**Occupancy guidelines**:
- **> 50%**: Generally good
- **> 75%**: Excellent for most kernels
- **100%**: Perfect, but not always necessary

**Trade-off**: Sometimes lower occupancy with more registers per thread is faster!

---

### 6. `launch_bounds_example.cu` - Control Resource Usage

**Purpose**: Use `__launch_bounds__` to optimize register and shared memory usage.

**Code**:
```cpp
__global__ void __launch_bounds__(256, 4)  // 256 threads, min 4 blocks/SM
myKernel(float* data) {
    // Compiler guarantees:
    // - Kernel uses ≤ registers to fit 4 blocks/SM
    // - Optimizes for 256 threads per block
}
```

**When to use**:
- Kernel has high register pressure → Limit registers
- Want guaranteed occupancy → Specify min blocks/SM
- Performance tuning → Experiment with different bounds

**How to run**:
```bash
make launch_bounds_example
./launch_bounds_example_sm100
```

---

### 7. `unified_memory.cu` - Managed Memory

**Purpose**: Simplify memory management with CUDA Unified Memory.

**Traditional CUDA** (manual management):
```cpp
float *h_data, *d_data;
h_data = (float*)malloc(size);
cudaMalloc(&d_data, size);
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
kernel<<<...>>>(d_data);
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
```

**Unified Memory** (automatic):
```cpp
float *data;
cudaMallocManaged(&data, size);  // Accessible from CPU and GPU!
kernel<<<...>>>(data);  // Automatic migration
// Use data on CPU directly, no copy needed!
```

**Benefits**:
- ✅ Simpler code (no explicit transfers)
- ✅ Automatic page migration
- ✅ Oversubscription (use more memory than GPU has)

**Drawbacks**:
- ❌ Slower than explicit transfers (page fault overhead)
- ❌ Less control over placement
- ❌ Not ideal for high-performance code

**When to use**: Prototyping, irregular access patterns, oversubscription scenarios.

**How to run**:
```bash
make unified_memory
./unified_memory_sm100
```

---

## Grid and Block Dimension Calculation

### 1D Problems

```cpp
int threads = 256;  // Typical: 128, 256, or 512
int blocks = (n + threads - 1) / threads;  // Ceiling division
kernel<<<blocks, threads>>>(data, n);
```

**Why 256 threads?**
- Multiple of warp size (32) ✅
- Good occupancy on most GPUs ✅
- Balances parallelism and resource usage ✅

### 2D Problems

```cpp
dim3 threads(16, 16);  // 256 threads total
dim3 blocks(
    (width + threads.x - 1) / threads.x,
    (height + threads.y - 1) / threads.y
);
kernel<<<blocks, threads>>>(data, width, height);
```

**Common 2D sizes**: (16,16), (32,8), (8,32) depending on access pattern.

### 3D Problems (rare)

```cpp
dim3 threads(8, 8, 8);  // 512 threads total
dim3 blocks(
    (depth + 7) / 8,
    (height + 7) / 8,
    (width + 7) / 8
);
```

**Use cases**: 3D convolutions, volumetric data, simulations.

---

## Performance Analysis

### Using Common Infrastructure

```bash
# Profile CUDA kernel
../../common/profiling/profile_cuda.sh ./add_parallel baseline
../../common/profiling/profile_cuda.sh ./add_sequential baseline

# Compare in Nsight Compute
ncu-ui ../../results/ch6/add_parallel_baseline_metrics_*.ncu-rep
```

### Key Metrics to Watch

| Metric | Target | How to Check |
|--------|--------|-------------|
| Occupancy | > 50% | Nsight Compute |
| Warp Execution Efficiency | > 90% | No divergence |
| Memory Throughput | Close to peak | Coalesced access (Ch7) |
| Compute Throughput | Varies | Depends on algorithm |

---

## How to Run All Examples

```bash
cd ch6

# Build all examples
make

# Run in order of complexity
./my_first_kernel_sm100          # Hello world
./add_sequential_sm100           # Baseline (slow)
./add_parallel_sm100             # Optimized (fast!)
./2d_kernel_sm100                # 2D indexing
./occupancy_api_sm100            # Check occupancy
./launch_bounds_example_sm100    # Resource control
./unified_memory_sm100           # Managed memory

# Profile for learning
../../common/profiling/profile_cuda.sh ./add_parallel baseline
```

---

## Key Takeaways

1. **Thread indexing is fundamental**: `idx = blockIdx.x * blockDim.x + threadIdx.x` is the most important pattern in CUDA.

2. **Parallelization gives massive speedups**: Even naive parallel code can be 100-1000x faster than sequential.

3. **Block size matters**: 256 threads/block is a good default. Tune for your specific kernel.

4. **Bounds checking is essential**: Always check `if (idx < n)` to avoid out-of-bounds access.

5. **Occupancy isn't everything**: 100% occupancy doesn't guarantee best performance. Memory access patterns (Ch7) often matter more.

6. **Start simple, optimize later**: Get correct parallel code first, then optimize (Chapters 7-10).

---

## Common Pitfalls

### Pitfall 1: Forgetting Bounds Check
**Problem**: Array out-of-bounds when `n` not divisible by block size.

**Solution**: Always add bounds check:
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {  // Critical!
    data[idx] = ...;
}
```

### Pitfall 2: Wrong Grid Size Calculation
**Problem**: Integer division truncates → some elements not processed.

**Wrong**:
```cpp
int blocks = n / threads;  // Truncates!
```

**Correct**:
```cpp
int blocks = (n + threads - 1) / threads;  // Ceiling division
```

### Pitfall 3: Not Synchronizing After Kernel
**Problem**: Accessing results before kernel completes.

**Solution**: Call `cudaDeviceSynchronize()` or check async:
```cpp
kernel<<<...>>>();
cudaDeviceSynchronize();  // Wait for completion
// Now safe to access results
```

### Pitfall 4: Using Too Few Threads
**Problem**: Block size 32 → Low occupancy → Poor performance.

**Solution**: Use 128-512 threads per block for good occupancy.

### Pitfall 5: Unified Memory for High-Performance Code
**Problem**: Using `cudaMallocManaged` in performance-critical code.

**Reality**: Explicit `cudaMemcpy` is faster. Use unified memory for prototyping only.

---

## Next Steps

**Continue CUDA mastery** → [Chapter 7: Memory Access Patterns](../ch7/README.md)

Learn about:
- Memory coalescing for 10x bandwidth improvement
- Shared memory for data reuse
- Bank conflicts and how to avoid them
- Vectorized memory access

**Back to PyTorch land** → [Chapter 13: PyTorch Profiling](../ch13/README.md)

---

## Additional Resources

- **CUDA C Programming Guide**: [Official NVIDIA Docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- **CUDA Best Practices**: [Performance Guidelines](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- **Nsight Compute**: [Profiler User Guide](https://docs.nvidia.com/nsight-compute/)
- **Common Headers**: See `../../common/headers/cuda_helpers.cuh` for error checking macros

---

**Chapter Status**: ✅ Complete  
**Last Updated**: November 3, 2025  
**Tested On**: NVIDIA B200 GPU, CUDA 13.0, sm_100 architecture

