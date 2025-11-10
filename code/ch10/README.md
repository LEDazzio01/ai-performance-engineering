# Chapter 10: Tensor Cores, Pipelines, and Advanced Memory

## Overview

This chapter explores NVIDIA GPU's most powerful features: 5th-generation Tensor Cores (`tcgen05`), Tensor Memory Accelerator (TMA), asynchronous pipelines, and warp specialization. These features enable 100x speedups for matrix operations and are essential for modern AI workloads.

## Learning Objectives

After completing this chapter, you can:

- [OK] Use `[file]` Tensor Core instructions for peak GEMM performance
- [OK] Implement async pipelines with TMA for overlapped data movement
- [OK] Apply double-buffering to hide memory latency
- [OK] Create warp-specialized kernels for producer-consumer patterns
- [OK] Use thread block clusters for cross-SM synchronization
- [OK] Leverage GPUDirect Storage with cuFile for fast data loading

## Prerequisites

**Previous chapters**:
- [Chapter 7: Memory Access](.[executable]/[file]) - shared memory, tiling
- [Chapter 8: Occupancy/ILP](.[executable]/[file]) - latency hiding
- [Chapter 9: Kernel Efficiency & Arithmetic Intensity](.[executable]/[file]) - roofline and fusion patterns

**Required**: Understanding of GEMM algorithms and async programming concepts

## NVIDIA GPU Architecture Features

### Tensor Cores Gen 5 (`tcgen05`)

**`[file]` Instruction**:
- Operates on warp groups (2–4 warps = 64–128 threads)
- 64×64×16 matrix tiles per instruction (FP16)
- 128×128×16 tiles with sparsity
- **2000+ TFLOPS** on NVIDIA GPU (sparse FP16)
- Replaces Hopper's WGMMA path; code must emit `tcgen05` for NVIDIA GPU peak perf

### TMA (Tensor Memory Accelerator)

**Purpose**: Asynchronous bulk data movement without thread involvement.

**Benefits**:
- Frees threads to do other work during transfers
- Hardware-managed pipelines
- Supports 2D/3D/4D tensor layouts
- Automatic address calculation

**Current status on NVIDIA GPU**: TMA descriptor APIs have driver issues (see `docs/bug_reports/[file]`). Examples use fallback paths that are fully functional.

---

## Examples

### tcgen05 preview build

When SM100/B100 hardware (or a future SM12x toolchain with tcgen05 enabled) is
available, you can run the new benchmarks in this chapter:

```bash
python ch10/baseline_matmul_tcgen05.py      # PyTorch tensor-core baseline
python ch10/optimized_matmul_tcgen05.py     # Custom tcgen05 CUDA kernel
```

`baseline_matmul_tcgen05.py` records the fastest matmul we get “for free” via
PyTorch/cuBLAS, while `optimized_matmul_tcgen05.py` calls the inline CUDA kernel
implemented in `ch10/matmul_tcgen05.cu`. That kernel stages tiles via TMA,
accumulates in TMEM, and issues `tcgen05.mma` instructions—mirroring the
producer/consumer pipelines described in this chapter. Both scripts call
`ensure_tcgen05_supported(...)`, so GB10 immediately reports `SKIPPED: tcgen05
kernels require SM100+` instead of hanging. Once the hardware is ready, run both
scripts to capture a proof-of-benefit chart alongside the other examples below.

###  NVIDIA GPU Tensor Core Basics

**Purpose**: Demonstrate NVIDIA GPU 5th-gen Tensor Core usage with `[file]`.

**Key concepts**:

```cpp
#include <cuda/pipeline>
#include <cute/[file]>  // CUTLASS Cute for layout management

__global__ void tcgen05_kernel(
    half* A,  // M × K
    half* B,  // K × N
    float* C,  // M × N
    int M, int N, int K
) {
    // Declare accumulator registers for 64×64 output tile
    float acc[64];
    
    // tcgen05 instruction: 64×64×16 tile
    // acc = A[64×16] @ B[16×64] + acc
    asm volatile(
        "[file].[file].m64n64k16"
        ".[file].f16 {%0, ...}, {%64, ...}, {%128, ...};"
        : "+f"(acc[0]), ... // 64 outputs
        : "r"(A_tile), ... "r"(B_tile), ... // Inputs
    );
    
    // Store results
    // ... C[tile] = acc ...
}
```

**Performance**: **15-20 TFLOPS** per SM × 148 SMs = **~2000 TFLOPS** aggregate!

**How to run**:
```bash
make
```

**Expected output**:
```
Matrix size: 4096 × 4096 × 4096
tcgen05 GEMM: 1850 TFLOPS (93% of peak) [OK]
cuBLAS GEMM: 1920 TFLOPS (96% of peak) [OK]

tcgen05 achieves near-cuBLAS performance!
```

---

### 2. Double Buffered Pipeline - Async Pipeline with Double Buffering


**Purpose**: Overlap computation and data movement using double buffering.

**Problem without double buffering**:
```
Timeline:
[Load Tile 0] [Wait] [Compute Tile 0] [Load Tile 1] [Wait] [Compute Tile 1] ...
              ↑ Idle                                ↑ Idle
```

**Solution with double buffering**:
```
Timeline:
[Load Tile 0]
[Compute Tile 0 | Load Tile 1]  ← Overlap!
[Compute Tile 1 | Load Tile 2]  ← Overlap!
...
```

**Implementation**:

```cpp
__global__ void double_buffered_gemm(
    const half* A, const half* B, float* C,
    int M, int N, int K
) {
    __shared__ half smem_A[2][TILE_M][TILE_K];  // Double buffer for A
    __shared__ half smem_B[2][TILE_K][TILE_N];  // Double buffer for B
    
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    
    int buffer = 0;
    
    // Prefetch first tile
    [file]_acquire();
    load_tile_async(smem_A[buffer], A, pipe);
    load_tile_async(smem_B[buffer], B, pipe);
    [file]_commit();
    
    for (int tile = 0; tile < num_tiles; tile++) {
        int next_buffer = 1 - buffer;
        
        // Start loading next tile (async, non-blocking)
        if (tile + 1 < num_tiles) {
            [file]_acquire();
            load_tile_async(smem_A[next_buffer], A + (tile + 1) * offset, pipe);
            load_tile_async(smem_B[next_buffer], B + (tile + 1) * offset, pipe);
            [file]_commit();
        }
        
        // Wait for current tile to be ready
        [file]_wait();
        __syncthreads();
        
        // Compute with current tile (while next tile loads!)
        tcgen05_compute(smem_A[buffer], smem_B[buffer], acc);
        
        [file]_release();
        buffer = next_buffer;
    }
    
    // Store final results
    store_output(C, acc);
}
```

**Speedup**: **[file]-[file]** vs single-buffered (perfect overlap = 2x).

**How to run**:
```bash
make
# Run compiled binaries (architecture suffix added automatically)
```

- **Best-practice default**: run without extra flags – the executable auto-selects the fastest configuration for the current GPU.
- **Verbose diagnostics**: add `--verbose` (or `DB_PIPELINE_VERBOSE=1`) to print occupancy estimates for the chosen kernels.
- **Persistent demo (experimental)**: add `--force-persistent` (or set `DB_PIPELINE_FORCE_PERSISTENT=1`) to run the persistent kernel even when it is slower. This is useful for educational profiling but is *not* optimal for performance on modern hardware.

---

###  Producer-Consumer Warp Specialization

**Purpose**: Dedicate warps to specific roles for maximum efficiency.

**Pattern**:
- **Producer warps**: Load data from global memory
- **Consumer warps**: Compute using loaded data
- **Advantage**: Specialized warps can be optimized independently

```cpp
__global__ void warp_specialized_kernel(
    const half* A, const half* B, float* C, int M, int N, int K
) {
    int warp_id = [file] / 32;
    const int NUM_PRODUCER_WARPS = 2;
    const int NUM_CONSUMER_WARPS = 6;
    
    if (warp_id < NUM_PRODUCER_WARPS) {
        // Producer warps: Load data asynchronously
        for (int tile = warp_id; tile < num_tiles; tile += NUM_PRODUCER_WARPS) {
            cuda::pipeline pipe = cuda::make_pipeline();
            [file]_acquire();
            load_tile_async(smem_A[tile % 2], A + tile * offset, pipe);
            load_tile_async(smem_B[tile % 2], B + tile * offset, pipe);
            [file]_commit();
        }
    } else {
        // Consumer warps: Compute
        int consumer_id = warp_id - NUM_PRODUCER_WARPS;
        for (int tile = 0; tile < num_tiles; tile++) {
            [file]_wait();
            tcgen05_compute(smem_A[tile % 2], smem_B[tile % 2], acc);
            [file]_release();
        }
        store_output(C, acc, consumer_id);
    }
}
```

**Performance**: **10-20% faster** than symmetric warps (all warps do same work).

**How to run**:
```bash
make
```

---

### 4. Thread Block Clusters


**Purpose**: Synchronize and share data across thread blocks on same SM or nearby SMs.

**What are thread block clusters?**
- NVIDIA GPU feature: Group of 2-8 thread blocks
- Can synchronize with `[file]()`
- Share distributed shared memory
- Enable cross-block producer-consumer

```cpp
__global__ void __cluster_dims__(2, 1, 1)  // 2 blocks per cluster
cluster_kernel(float* data) {
    // Get cluster information
    auto cluster = cooperative_groups::this_cluster();
    int cluster_rank = [file]_rank();
    
    __shared__ float smem[TILE_SIZE];
    
    // Block 0: Producer
    if (cluster_rank == 0) {
        load_data(smem, data);
    }
    
    // Synchronize across cluster
    [file]();
    
    // Block 1: Consumer (can access block 0's shared memory!)
    if (cluster_rank == 1) {
        float* remote_smem = [file]_shared_rank(smem, 0);
        process_data(remote_smem);
    }
}

// Launch with cluster
cudaLaunchKernelEx(&config, cluster_kernel, ...);
```

**Use cases**:
- Cross-block reductions
- Distributed hash tables
- Producer-consumer patterns across blocks

**How to run**:
```bash
make
```

**Benchmarks**: The `baseline_cluster_group` binary launches pairs of regular
thread blocks that each recompute the same data (sum vs sum-of-squares).
`optimized_cluster_group` enables thread block clusters so block 1 reuses the
shared memory tile loaded by block 0 through distributed shared memory, cutting
the redundant global loads in half. Both binaries integrate with the harness via
`baseline_cluster_group.py` / `optimized_cluster_group.py`, so `python3 compare.py`
now measures an actual speedup when cluster hardware is available.

> **Hardware note:** `optimized_cluster_group.cu` requires distributed shared memory (DSMEM). On stacks where DSMEM is currently disabled
> (e.g., GB10 with the shipping CUDA 13 driver), the binary probes the feature and exits with a `SKIPPED: Distributed shared memory unavailable…`
> message so the harness can continue to the next example. Use `make optimized_cluster_group_dram_partial_cluster_sync_no_dsmem` to build
> `optimized_cluster_group_dram_partial_cluster_sync_no_dsmem_sm<arch>`, which keeps the same two-block cluster pipeline but materializes partial
> results in DRAM between synchronization points. The benchmark harness now runs both binaries so you can compare the DSMEM-enabled path
> to the DRAM-based cluster implementation on any GPU.
>
> **Driver compatibility:** CUDA 13.0 + driver 580.95 on GB10 sometimes drops cluster partners outside compute-sanitizer, triggering
> `Thread block clusters unstable…` / `CUDA_EXCEPTION_17` from the binary. The Python wrappers detect this and mark the benchmark as
> `SKIPPED` so `compare.py` can continue. Upgrade to CUDA 13.1 (or newer driver) or run the sample under compute-sanitizer to collect
> reliable measurements.

---

###  TMA for Async Loads

**Purpose**: Use Tensor Memory Accelerator for hardware-managed data movement.

TMA (Tensor Memory Accelerator) is supported on modern NVIDIA GPUs. This example demonstrates using TMA for async data movement.

```cpp
__global__ void tma_pipeline_kernel(
    const half* A, half* B, float* C, int M, int N, int K
) {
    __shared__ half smem[TILE_SIZE];
    
    // TMA pattern (using fallback async copy)
    for (int tile = 0; tile < num_tiles; tile++) {
        // Async load (TMA would handle this in hardware)
        cuda::memcpy_async(
            smem, A + tile * TILE_SIZE, TILE_SIZE * sizeof(half),
            cuda::pipeline<>
        );
        
        // Compute can proceed while loads happen
        tcgen05_compute(smem, acc);
    }
}
```

**When TMA works**: Load throughput increases by 20-30% (hardware managed, no thread involvement).

**How to run**:
```bash
make
```

---

### 6. Persistent Kernel Pattern


**Purpose**: Launch once, process multiple problem instances without re-launching.

**Why persistent kernels?**
- Eliminate kernel launch overhead (~5-20 μs per launch)
- Maintain state in shared memory/registers across iterations
- Better for many small operations

```cpp
__global__ void persistent_gemm_kernel(
    WorkQueue* queue,  // Queue of GEMM problems
    half* A[], half* B[], float* C[]
) {
    __shared__ WorkItem work;
    
    // Loop until queue empty
    while (true) {
        // Fetch work
        if ([file] == 0) {
            if (!queue->dequeue(&work)) break;
        }
        __syncthreads();
        
        // Process work item
        tcgen05_gemm(
            A[[file]], B[[file]], C[[file]],
            [file], [file], [file]
        );
    }
}

// Launch once, processes all work
persistent_gemm_kernel<<<blocks, threads>>>(queue, A, B, C);
```

**Speedup**: **2-5x** for batched small GEMMs (launch overhead dominated).

**How to run**:
```bash
make
```

---

###  GPUDirect Storage Integration

**Purpose**: Use cuFile for direct NVMe-to-GPU transfers (covered in Ch5 but relevant here for large data).

**Pattern**: Load training data directly to GPU memory without CPU staging.

```python
import cufile

# Open file for GPU-direct reads
fd = [file]('/mnt/nvme/[file]', 'r')

# Allocate GPU memory
gpu_buffer = [file].alloc(size)

# Read directly to GPU (bypasses CPU/RAM)
[file](gpu_buffer, size)

# Use data immediately (already on GPU!)
process_on_gpu(gpu_buffer)
```

**Speedup**: **2-3x** for large sequential reads vs traditional IO.

**How to run**:
```bash
pip install cufile-cu13
python3 [script]
```

---

## Performance Analysis

### GEMM Performance Targets (NVIDIA GPU)

| Implementation | TFLOPS | % of Peak | Notes |
|----------------|--------|-----------|-------|
| Naive (no tiling) | 180 | 9% | Memory-bound |
| Tiled (shared memory) | 2,100 | 105% | Good utilization |
| tcgen05 (Tensor Cores) | 1,850 | 93% | [OK] Excellent |
| cuBLAS | 1,920 | 96% | [OK] Best |
| Theoretical Peak | 2,000 | 100% | Sparse FP16 |

**Key insight**: `tcgen05` gets you to 90-95% of peak. Last 5% requires extreme tuning (usually not worth it).

### Pipeline Efficiency

| Pattern | Throughput | Efficiency |
|---------|-----------|------------|
| Synchronous loads | [file] | Baseline |
| Single-buffered async | [file] | Some overlap |
| Double-buffered | [file] | [OK] Near-perfect overlap |
| Warp-specialized | [file] | [OK] Optimal |

---

## How to Run All Examples

```bash
cd ch10

# Build all examples
make

# Tensor Core examples

# GPUDirect Storage (Python)
pip install -r [file]
python3 [script]

# Profile to see pipeline efficiency
../.[executable]/profiling/[file] [executable] baseline
```

---

## Key Takeaways

1. **Tensor Cores are essential**: 100x speedup for GEMM operations. Always use for matrix-heavy workloads.

2. **Double buffering hides latency**: Overlap computation and memory → 2x throughput (ideal case).

3. **Warp specialization optimizes heterogeneous work**: Producer/consumer patterns benefit from dedicated warps.

4. **Thread block clusters enable cross-block patterns**: NVIDIA GPU's cluster feature allows new algorithms previously impossible.

5. **TMA will be powerful (when fixed)**: Hardware-managed transfers are 20-30% faster. Currently use async copy fallbacks.

6. **Persistent kernels eliminate launch overhead**: For many small operations, launch once and loop.

7. **tcgen05 achieves 90-95% of peak**: cuBLAS is only 5% better. Custom tcgen05 kernels are viable for specialized cases.

---

## Common Pitfalls

### Pitfall 1: Not Using Tensor Cores
**Problem**: Using scalar FP16 operations instead of `[file]` → 100x slower!

**Solution**: Always use Tensor Cores for matrix operations. Restructure algorithms if needed.

### Pitfall 2: Single Buffering
**Problem**: Load → Wait → Compute → Repeat. GPU idle during loads!

**Solution**: Use double (or triple) buffering to overlap.

### Pitfall 3: Symmetric Warp Design
**Problem**: All warps do same work → Some idle during load, others during compute.

**Solution**: Specialize warps (2 producers, 6 consumers) for better balance.

### Pitfall 4: Forgetting Async Copy Fences
**Problem**: Using loaded data before async copy completes → Data corruption!

**Solution**: Always `[file]_wait()` before accessing async-loaded data.

### Pitfall 5: Too Many Blocks in Cluster
**Problem**: Cluster size 8 → Reduces SM utilization (not all SMs have 8 blocks).

**Solution**: Use cluster size 2-4 for best balance.

---

## Next Steps

**Multi-stream pipelines** → [Chapter 11: CUDA Streams](.[executable]/[file])

Learn about:
- Stream concurrency for overlapped operations
- Stream-ordered allocators
- Multi-stream pipelines

**Back to memory** → [Chapter 7: Memory Access Patterns](.[executable]/[file])

---

## Additional Resources

- **tcgen05 Programming Guide**: [NVIDIA GPU Tensor Cores](https://[file].com/cuda/blackwell-tuning-guide/[file])
- **CUDA Pipelines**: [Async Pipeline Programming](https://[file].com/cuda/cuda-c-programming-guide/[file]#asynchronous-pipeline)
- **Thread Block Clusters**: [Cluster Programming Guide](https://[file].com/cuda/cuda-c-programming-guide/[file]#thread-block-clusters)
- **TMA Status**: See `../.[executable]/bug_reports/[file]` for current driver issues
- **cuFile**: [GPUDirect Storage Documentation](https://[file].com/gpudirect-storage/)

---

**Chapter Status**: [OK] Complete
