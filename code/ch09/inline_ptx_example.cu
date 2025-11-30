// Architecture-specific optimizations for CUDA 13.0
// Targets Blackwell B200/B300 (sm_100)
// inline_ptx_example.cu
// Example demonstrating inline PTX for micro-optimizations

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <stdio.h>
#include <cmath>
#include <vector>

#include "../core/common/headers/tma_helpers.cuh"

static void run_tma_example();

static void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        printf("CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

// Example kernel using inline PTX for prefetching
__global__ void PrefetchExample(const float *in, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // The runtime has no device-side builtin for L2 prefetch; use cp.async.bulk.prefetch.
        // Prefetch the next cache line (128B) of in[] into L2:
        if (idx + 32 < N) {
            const float *next_ptr = in + idx + 32;
            const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(next_ptr);
            if ((addr & 0x7F) == 0) {  // issue only when 128B aligned to avoid misaligned traps
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                             :
                             : "l"(next_ptr), "n"(128));
            }
        }
        
        float x = in[idx];
        
        // Do some work here before using in[idx+32] to give time for prefetch
        float result = x;
        for (int i = 0; i < 10; ++i) {
            result = result * 1.1f + 0.1f;
        }
        
        out[idx] = result;
    }
}

// Example using inline PTX for cache control
__global__ void CacheControlExample(const float *in, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float val;
        // Load with cache global (.cg) modifier - cache in L2, bypass L1
        asm("ld.global.cg.f32 %0, [%1];" : "=f"(val) : "l"(in + idx));
        
        // Process the value
        val = val * val + 1.0f;
        
        out[idx] = val;
    }
}

// Example using inline PTX to read special registers
__global__ void SpecialRegisterExample(int *smid_output, int *lane_output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Get SM ID using inline PTX
    unsigned int smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    
    // Get lane ID (warp-local thread index)
    unsigned int laneid;
    asm("mov.u32 %0, %laneid;" : "=r"(laneid));
    
    smid_output[idx] = smid;
    lane_output[idx] = laneid;
}

// Example demonstrating manual instruction scheduling with PTX
__global__ void InstructionSchedulingExample(const float *a, const float *b, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N && idx + 1 < N) {
        float val1, val2;
        
        // Manual scheduling: issue both loads back-to-back to overlap latencies
        asm("ld.global.f32 %0, [%1];" : "=f"(val1) : "l"(a + idx));
        asm("ld.global.f32 %0, [%1];" : "=f"(val2) : "l"(b + idx + 1));
        
        // Now compute on both values
        float result = val1 * val2 + val1 + val2;
        
        out[idx] = result;
    }
}

int main() {
    const int N = 1024 * 1024;
    size_t bytes = N * sizeof(float);
    
    // Allocate host memory
    float *h_in = new float[N];
    float *h_out = new float[N];
    int *h_smid = new int[N];
    int *h_lane = new int[N];
    
    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_in[i] = float(i % 1000) / 1000.0f;
    }
    
    // Allocate device memory
    float *d_in, *d_out, *d_b;
    int *d_smid, *d_lane;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_smid, N * sizeof(int));
    cudaMalloc(&d_lane, N * sizeof(int));
    
    // Copy input to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_in, bytes, cudaMemcpyHostToDevice);
    
    // Launch parameters
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    printf("=== Inline PTX Examples ===\n");
    
    // Test prefetch example
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    PrefetchExample<<<blocks, threads>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError(), "PrefetchExample");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Prefetch example: %.2f ms\n", ms);
    
    // Test cache control example
    cudaEventRecord(start);
    CacheControlExample<<<blocks, threads>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError(), "CacheControlExample");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&ms, start, stop);
    printf("Cache control example: %.2f ms\n", ms);
    
    // Test special register example
    SpecialRegisterExample<<<blocks, threads>>>(d_smid, d_lane);
    checkCuda(cudaDeviceSynchronize(), "SpecialRegisterExample");
    
    // Copy results back
    cudaMemcpy(h_smid, d_smid, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lane, d_lane, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Special registers example:\n");
    printf("  Thread 0: SM ID = %d, Lane ID = %d\n", h_smid[0], h_lane[0]);
    printf("  Thread 32: SM ID = %d, Lane ID = %d\n", h_smid[32], h_lane[32]);
    printf("  Thread 64: SM ID = %d, Lane ID = %d\n", h_smid[64], h_lane[64]);
    
    // Test instruction scheduling example
    cudaEventRecord(start);
    InstructionSchedulingExample<<<blocks, threads>>>(d_in, d_b, d_out, N);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError(), "InstructionSchedulingExample");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&ms, start, stop);
    printf("Instruction scheduling example: %.2f ms\n", ms);
    
    // Copy final result
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printf("Final result[0]: %.3f\n", h_out[0]);
    
    printf("\n=== PTX Optimization Notes ===\n");
    printf("1. cp.async.bulk.prefetch.L2 - Manually prefetch data into L2 cache\n");
    printf("2. ld.global.cg - Load with cache global hint (L2 only, bypass L1)\n");
    printf("3. %%smid, %%laneid - Special registers for SM and lane identification\n");
    printf("4. Manual scheduling - Issue independent loads back-to-back for ILP\n");
    
    printf("\nTo analyze with Nsight Compute:\n");
    printf("ncu --section MemoryWorkloadAnalysis --section WarpStateStats ./inline_ptx_example\n");

    // Run the TMA example (Blackwell/Hopper+)
    run_tma_example();
    
    // Cleanup
    delete[] h_in;
    delete[] h_out;
    delete[] h_smid;
    delete[] h_lane;
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_b);
    cudaFree(d_smid);
    cudaFree(d_lane);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

// CUDA 13.0 Stream-ordered Memory Allocation Example
__global__ void stream_ordered_memory_example() {
    // Example of stream-ordered memory allocation
    // This is a placeholder for actual implementation
    // See ch11/stream_ordered_allocator.cu for a full cudaMallocAsync demo.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Your kernel code here
}

#if CUDART_VERSION >= 13000
namespace cde = cuda::device::experimental;

template <int TILE_SIZE>
__global__ void tma_example_kernel(const __grid_constant__ CUtensorMap in_desc,
                                   const __grid_constant__ CUtensorMap out_desc,
                                   int total_tiles) {
    constexpr std::size_t BYTES_PER_TILE = static_cast<std::size_t>(TILE_SIZE) * sizeof(float);
    __shared__ alignas(128) float stage_buffers[2][TILE_SIZE];
    using block_barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(block_barrier) unsigned char barrier_storage[2][sizeof(block_barrier)];

    auto init_barrier = [](block_barrier* bar) {
        init(bar, blockDim.x);
    };

    if (threadIdx.x == 0) {
        for (int i = 0; i < 2; ++i) {
            init_barrier(reinterpret_cast<block_barrier*>(barrier_storage[i]));
        }
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    cuda::barrier<cuda::thread_scope_block>::arrival_token tokens[2];

    auto issue_tile = [&](int tile_idx, int local_seq) {
        if (tile_idx >= total_tiles) {
            return;
        }
        const int stage = local_seq % 2;
        auto* bar_ptr = reinterpret_cast<block_barrier*>(barrier_storage[stage]);
        auto& bar = *bar_ptr;

        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_1d_global_to_shared(
                &stage_buffers[stage],
                &in_desc,
                tile_idx * TILE_SIZE,
                bar);
            tokens[stage] = cuda::device::barrier_arrive_tx(
                bar,
                1,
                BYTES_PER_TILE);
        } else {
            tokens[stage] = bar.arrive();
        }
    };

    const int base_tile = static_cast<int>(blockIdx.x);
    const int stride_tiles = static_cast<int>(gridDim.x);
    const int tiles_this_block = (total_tiles <= base_tile)
                                     ? 0
                                     : (total_tiles - base_tile + stride_tiles - 1) / stride_tiles;

    const int preload = (tiles_this_block < 2) ? tiles_this_block : 2;
    for (int t = 0; t < preload; ++t) {
        issue_tile(base_tile + t * stride_tiles, t);
    }

    for (int local_seq = 0; local_seq < tiles_this_block; ++local_seq) {
        const int stage = local_seq % 2;
        auto* bar_ptr = reinterpret_cast<block_barrier*>(barrier_storage[stage]);
        auto& bar = *bar_ptr;

        bar.wait(std::move(tokens[stage]));
        __syncthreads();

        float* tile_ptr = stage_buffers[stage];
        for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
            tile_ptr[i] = tile_ptr[i] * 1.5f + 1.0f;
        }
        cde::fence_proxy_async_shared_cta();
        __syncthreads();

        if (threadIdx.x == 0) {
            const int global_tile = base_tile + local_seq * stride_tiles;
            cde::cp_async_bulk_tensor_1d_shared_to_global(
                &out_desc,
                global_tile * TILE_SIZE,
                &stage_buffers[stage]);
            cde::cp_async_bulk_commit_group();
            cde::cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();

        const int next_seq = local_seq + 2;
        if (next_seq < tiles_this_block) {
            const int next_global = base_tile + next_seq * stride_tiles;
            issue_tile(next_global, next_seq);
        }
    }
}

static void run_tma_example() {
    if (!cuda_tma::device_supports_tma()) {
        printf("TMA example: SKIPPED (no Blackwell/Hopper-class TMA support)\n");
        return;
    }

    constexpr int TILE = 256;
    constexpr int ELEMS = 1 << 22;  // ~4M elems to better saturate TMA path
    constexpr int THREADS = 256;
    const auto limits = cuda_arch::get_tma_limits();
    if (TILE > static_cast<int>(limits.max_1d_box_size)) {
        printf("TMA example: SKIPPED (tile=%d exceeds 1D box limit=%u)\n",
               TILE,
               static_cast<unsigned int>(limits.max_1d_box_size));
        return;
    }

    std::vector<float> h_in(ELEMS);
    for (int i = 0; i < ELEMS; ++i) {
        h_in[i] = static_cast<float>((i % 512) - 256) * 0.25f;
    }
    std::vector<float> h_out(ELEMS, 0.0f);

    float* d_in = nullptr;
    float* d_out = nullptr;
    cudaGetLastError();  // clear any lingering errors from earlier kernels
    checkCuda(cudaMalloc(&d_in, ELEMS * sizeof(float)), "cudaMalloc d_in");
    checkCuda(cudaMalloc(&d_out, ELEMS * sizeof(float)), "cudaMalloc d_out");
    checkCuda(cudaMemcpy(d_in, h_in.data(), ELEMS * sizeof(float), cudaMemcpyHostToDevice), "copy h_in");
    checkCuda(cudaMemset(d_out, 0, ELEMS * sizeof(float)), "memset d_out");

    CUtensorMap in_desc{};
    CUtensorMap out_desc{};
    auto encode = cuda_tma::load_cuTensorMapEncodeTiled();
    if (!encode) {
        printf("TMA example: SKIPPED (cuTensorMapEncodeTiled unavailable)\n");
        cudaFree(d_in);
        cudaFree(d_out);
        return;
    }
    if (!cuda_tma::make_1d_tensor_map(in_desc, encode, d_in, ELEMS, TILE) ||
        !cuda_tma::make_1d_tensor_map(out_desc, encode, d_out, ELEMS, TILE)) {
        printf("TMA example: SKIPPED (descriptor creation failed)\n");
        cudaFree(d_in);
        cudaFree(d_out);
        return;
    }

    const int total_tiles = (ELEMS + TILE - 1) / TILE;
    int device = 0;
    cudaDeviceProp prop{};
    checkCuda(cudaGetDevice(&device), "get device");
    checkCuda(cudaGetDeviceProperties(&prop, device), "get device props");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int max_blocks = std::max(1, prop.multiProcessorCount * 16);
    const int blocks = std::min(total_tiles, max_blocks);

    tma_example_kernel<TILE><<<blocks, THREADS>>>(in_desc, out_desc, total_tiles);
    cudaDeviceSynchronize();

    constexpr int kIters = 10;
    cudaEventRecord(start);
    for (int i = 0; i < kIters; ++i) {
        tma_example_kernel<TILE><<<blocks, THREADS>>>(in_desc, out_desc, total_tiles);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    const float avg_ms = elapsed_ms / static_cast<float>(kIters);

    cudaMemcpy(h_out.data(), d_out, ELEMS * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < ELEMS; ++i) {
        const float expected = h_in[i] * 1.5f + 1.0f;
        if (std::abs(h_out[i] - expected) > 1e-3f) {
            printf("TMA example mismatch at %d: got %f expected %f\n", i, h_out[i], expected);
            ok = false;
            break;
        }
    }

    printf("TMA example: %.4f ms (avg over %d iters) [%s]\n",
           avg_ms, kIters, ok ? "OK" : "FAIL");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
}
#else
static void run_tma_example() {
    printf("TMA example: SKIPPED (requires CUDA 13+).\n");
}
#endif
