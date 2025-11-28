# Lab: Matching cuBLAS on Blackwell

## Overview

This **self-contained** lab demonstrates progressive, compounding optimizations 
of a tcgen05 GEMM kernel towards matching NVIDIA's cuBLAS on Blackwell B200 GPUs.

Each stage builds on the previous one, showing how optimizations compound.

**No dependencies on other chapters** - everything needed is in this directory.

## Results Summary

### At 16384×16384 (Large Workload) - Best for demonstrating optimizations

| Stage | Optimization | TFLOPS | % of cuBLAS | Improvement |
|-------|--------------|--------|-------------|-------------|
| 0 | cuBLAS (target) | **268** | 100% | — |
| 2 | + Tensor Cores (tcgen05) | 70 | 26% | baseline |
| 3 | + 2-Stage Pipeline | 95 | **36%** | **+1.36x** |
| 4 | + 3-Stage Pipeline | 119 | **45%** | **+1.25x** |
| 5 | + Swizzled Scheduling | 122 | **46%** | +1.03x |
| 6 | + Cluster Structure | 90 | 34% | (baseline for clusters) |
| 7 | Auto-Select Best | 122 | **46%** | picks best |

### At 4096×4096 (Smaller Workload)

| Stage | Optimization | TFLOPS | % of cuBLAS |
|-------|--------------|--------|-------------|
| 0 | cuBLAS | 207 | 100% |
| 4 | 3-Stage Pipeline | 80 | 39% |
| 5 | Swizzled | 80 | 39% |

**Key insight**: Larger matrices show better relative performance due to:
- Higher compute-to-memory ratio
- Better amortization of kernel launch overhead
- More tiles for pipeline to fill

## The Gap Analysis

We achieved **46% of cuBLAS** performance at 16K. The remaining **54% gap** comes from:

### What cuBLAS Does That We Don't

1. **Persistent Kernels** (~15-20% of gap)
   - CTAs stay resident and process multiple tiles
   - Amortizes kernel launch overhead
   - Better L2 cache locality between tiles
   - We: Launch new CTAs per tile

2. **Warp Specialization** (~10-15% of gap)
   - Dedicated producer warps (TMA loads)
   - Dedicated consumer warps (MMA compute)
   - Perfect overlap between memory and compute
   - We: All warps do both roles

3. **True Cluster Launch** (~5-10% of gap)
   - `cudaLaunchKernelEx` with cluster dimensions
   - TMA multicast to multiple CTAs in cluster
   - We: Cluster-ready code but regular launch

4. **SASS-level Optimization** (~10% of gap)
   - Hand-tuned instruction scheduling
   - Register allocation optimization
   - We: Compiler-generated code

5. **Epilogue Fusion** (~5% of gap)
   - Optimized TMEM→register→global path
   - Vectorized stores
   - We: Basic epilogue pattern

## Running the Lab

```bash
# Run all stages (skip naive for speed)
python run_lab.py --no-naive

# Run with large matrix (recommended for best results)
python run_lab.py --size 16384 --no-naive

# Run specific stage
python run_lab.py --stage 4 --size 16384

# Verify correctness
python run_lab.py --verify
```

## Files

- `run_lab.py` - Main lab runner with all stages
- `tcgen05_loader.py` - JIT compiler for CUDA kernels
- `tcgen05_gemm.cu` - Stage 2: Basic tcgen05 kernel
- `tcgen05_pipelined.cu` - Stage 3: 2-stage pipeline
- `tcgen05_3stage.cu` - Stage 4: 3-stage pipeline
- `tcgen05_swizzled.cu` - Stage 5: Swizzled tile scheduling
- `tcgen05_cluster.cu` - Stage 6: Cluster-optimized structure
- `autotune.py` - Stage 7: Auto-selection of best kernel
- `kernels.cu` - Stage 1: Naive SMEM kernel

## Key Concepts Demonstrated

### Stage 2: Tensor Cores (tcgen05)
- SM100_MMA_F16BF16_SS operation (128×256 tiles)
- TMA (Tensor Memory Accelerator) for async loads
- TMEM (Tensor Memory) for accumulator storage

### Stage 3: 2-Stage Pipeline
- Double-buffered shared memory
- Overlap TMA load of tile K+1 with compute of tile K

### Stage 4: 3-Stage Pipeline
- Triple-buffered shared memory
- Prefetch 2 tiles ahead
- Better latency hiding for long K dimension
- **45% of cuBLAS at 16K!**

### Stage 5: Swizzled Scheduling
- XOR swizzle pattern for tile ordering
- Improves L2 cache hit rate
- Reduces inter-SM contention

### Stage 6: Cluster Structure
- Code structured for thread block clusters
- Ready for `cudaLaunchKernelEx` cluster launch
- Foundation for TMA multicast

### Stage 7: Auto-Selection
- Benchmarks all kernels
- Caches optimal choice per problem size
- Adapts to hardware variations

## To Go Further (Reaching 70-90%)

To close the remaining gap to cuBLAS:

1. **Implement Persistent Kernels**
   - Global work counter for tile fetching
   - TMEM reused across tiles (already allocated)
   - Expected gain: +10-15%

2. **Add Warp Specialization**
   - Separate producer/consumer warp roles
   - `cuda::pipeline` for fine-grained sync
   - Expected gain: +10-15%

3. **Enable True Cluster Launch**
   - Use `cudaLaunchKernelEx` with cluster dims
   - Enable TMA multicast
   - Expected gain: +5-10%

4. **Study CUTLASS 4.x**
   - Look at `examples/cute/blackwell`
   - Reference SM100 GEMM implementations

## Requirements

- NVIDIA B200 or newer (SM 10.0+)
- CUDA 13.0+
- PyTorch 2.x with CUDA support
- CUTLASS 4.x (included in third_party/)
