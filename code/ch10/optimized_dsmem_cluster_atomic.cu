// optimized_dsmem_cluster_atomic.cu - DSMEM Cluster Atomic Reduction (Ch10)
//
// CHAPTER 10 CONTEXT: "Tensor Core Pipelines & Cluster Features"
// 
// KEY PATTERN: Cluster-wide atomic aggregation via DSMEM
//   1. Each CTA performs block-level reduction
//   2. Each CTA atomically adds to cluster leader's smem via map_shared_rank()
//   3. Cluster leader writes single result to global memory
//
// WHY THIS IS FASTER than two-pass reduction:
//   - No intermediate global memory writes between passes
//   - Cluster sync is cheaper than kernel launch overhead
//   - Single atomic per CTA instead of per-block global atomic

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

constexpr int BLOCK_SIZE = 256;
constexpr int CLUSTER_SIZE = 4;  // 4 CTAs per cluster
constexpr int ELEMENTS_PER_BLOCK = 4096;

//============================================================================
// Warp-level reduction using shuffle
//============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

//============================================================================
// Block-level reduction
//============================================================================

__device__ float block_reduce_sum(float val, float* smem) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;
    
    val = warp_reduce_sum(val);
    
    if (lane_id == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }
    
    return val;
}

//============================================================================
// DSMEM Cluster Reduction Kernel - NO FALLBACK
// Requires SM 9.0+ (Hopper/Blackwell)
//============================================================================

__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1)
__launch_bounds__(BLOCK_SIZE, 1)
void dsmem_cluster_reduction_kernel_v1(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int elements_per_cluster
) {
    // DSMEM is only available on SM 9.0+
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cg::thread_block block = cg::this_thread_block();
    cg::cluster_group cluster = cg::this_cluster();
    
    const int cluster_id = blockIdx.x / CLUSTER_SIZE;
    const int cluster_rank = cluster.block_rank();
    const int tid = threadIdx.x;
    
    // Shared memory for reductions
    __shared__ float smem_reduce[32];
    __shared__ float smem_cluster_sum;  // For cluster-level atomic aggregation
    
    // Global offset
    const int cluster_offset = cluster_id * elements_per_cluster;
    const int block_offset = cluster_offset + cluster_rank * ELEMENTS_PER_BLOCK;
    
    // Initialize cluster accumulator (only leader does this)
    if (cluster_rank == 0 && tid == 0) {
        smem_cluster_sum = 0.0f;
    }
    cluster.sync();
    
    //========================================================================
    // STEP 1: Each CTA reduces its chunk
    //========================================================================
    float local_sum = 0.0f;
    
    #pragma unroll 4
    for (int i = tid; i < ELEMENTS_PER_BLOCK; i += BLOCK_SIZE) {
        int global_idx = block_offset + i;
        if (global_idx < N) {
            local_sum += input[global_idx];
        }
    }
    
    float block_sum = block_reduce_sum(local_sum, smem_reduce);
    
    //========================================================================
    // STEP 2: Aggregate via DSMEM atomic to leader's shared memory
    //========================================================================
    if (tid == 0) {
        // Get pointer to leader's (rank 0) shared memory via DSMEM
        float* leader_sum = cluster.map_shared_rank(&smem_cluster_sum, 0);
        atomicAdd(leader_sum, block_sum);
    }
    
    // Synchronize entire cluster
    cluster.sync();
    
    //========================================================================
    // STEP 3: Cluster leader writes final result to global memory
    //========================================================================
    if (cluster_rank == 0 && tid == 0) {
        output[cluster_id] = smem_cluster_sum;
    }
    
#else
    // For older architectures, use a simple atomic fallback
    // This is not DSMEM but demonstrates the reduction pattern
    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * ELEMENTS_PER_BLOCK;
    
    __shared__ float smem_reduce[32];
    
    float local_sum = 0.0f;
    for (int i = tid; i < ELEMENTS_PER_BLOCK; i += BLOCK_SIZE) {
        int global_idx = block_offset + i;
        if (global_idx < N) {
            local_sum += input[global_idx];
        }
    }
    
    float block_sum = block_reduce_sum(local_sum, smem_reduce);
    
    if (tid == 0) {
        atomicAdd(&output[blockIdx.x / CLUSTER_SIZE], block_sum);
    }
#endif
}

//============================================================================
// Main - Benchmark with proper cluster launch
//============================================================================

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("DSMEM Cluster Reduction (Chapter 10 - NO FALLBACK)\n");
    printf("==================================================\n");
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    
    // FAIL FAST if DSMEM not supported
    if (prop.major < 9) {
        fprintf(stderr, "\nERROR: DSMEM requires SM 9.0+ (Hopper/Blackwell)\n");
        fprintf(stderr, "Device has SM %d.%d - use optimized_atomic_reduction instead\n",
                prop.major, prop.minor);
        return 1;
    }
    
    int cluster_supported = 0;
    cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch, 0);
    if (!cluster_supported) {
        fprintf(stderr, "\nERROR: Device does not support cluster launch\n");
        return 1;
    }
    printf("Cluster launch: SUPPORTED\n");
    
    // Problem size
    const int N = 16 * 1024 * 1024;
    const int elements_per_cluster = ELEMENTS_PER_BLOCK * CLUSTER_SIZE;
    const int num_clusters = (N + elements_per_cluster - 1) / elements_per_cluster;
    const int num_blocks = num_clusters * CLUSTER_SIZE;
    
    printf("\nProblem: %d elements (%.1f MB)\n", N, N * sizeof(float) / 1e6);
    printf("Clusters: %d (%d CTAs each)\n\n", num_clusters, CLUSTER_SIZE);
    
    // Allocate
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, num_clusters * sizeof(float)));
    
    // Initialize
    std::vector<float> h_input(N, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Configure cluster launch
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(num_blocks, 1, 1);
    config.blockDim = dim3(BLOCK_SIZE, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = 0;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_SIZE;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.numAttrs = 1;
    config.attrs = attrs;
    
    const int N_param = N;
    const int elements_param = elements_per_cluster;
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        CUDA_CHECK(cudaMemset(d_output, 0, num_clusters * sizeof(float)));
        CUDA_CHECK(cudaLaunchKernelEx(&config, 
                                       dsmem_cluster_reduction_kernel_v1,
                                       d_input, d_output, N_param, elements_param));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    const int iterations = 50;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaMemset(d_output, 0, num_clusters * sizeof(float)));
        CUDA_CHECK(cudaLaunchKernelEx(&config,
                                       dsmem_cluster_reduction_kernel_v1,
                                       d_input, d_output, N_param, elements_param));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    // Verify
    std::vector<float> h_output(num_clusters);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_clusters * sizeof(float), cudaMemcpyDeviceToHost));
    float total = std::accumulate(h_output.begin(), h_output.end(), 0.0f);
    
    printf("DSMEM Cluster Reduction:\n");
    printf("  Time: %.3f ms\n", ms / iterations);
    printf("  Sum: %.0f (expected: %d) - %s\n", total, N,
           (abs(total - N) < 1000) ? "PASS" : "FAIL");
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return 0;
}
