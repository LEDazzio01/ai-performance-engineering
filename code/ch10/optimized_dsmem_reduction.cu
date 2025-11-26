// optimized_dsmem_reduction.cu - Cross-CTA Reduction via DSMEM (Ch10)
//
// WHAT: Hierarchical reduction using Distributed Shared Memory (DSMEM)
// to aggregate results across CTAs in a thread block cluster.
//
// WHY: Traditional reductions require:
//   1. Per-block reduction to shared memory
//   2. Global memory write
//   3. Second kernel for final reduction
//
// With DSMEM:
//   1. Per-block reduction to shared memory
//   2. Cross-CTA reduction via DSMEM (no global memory!)
//   3. Single atomic write from cluster leader
//
// WHEN TO USE:
//   - Reductions where cluster CTAs have partial results
//   - Attention softmax normalization across sequence chunks
//   - MoE expert aggregation
//   - Any operation requiring cross-CTA communication
//
// REQUIREMENTS:
//   - SM 90+ (Hopper/Blackwell)
//   - Thread block clusters enabled
//   - Sufficient shared memory for reduction buffers

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
// Warp-level reduction
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
    
    // Warp reduction
    val = warp_reduce_sum(val);
    
    // Write warp results to shared memory
    if (lane_id == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    
    // First warp reduces all warp results
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }
    
    return val;
}

//============================================================================
// DSMEM Cluster Reduction Kernel
//============================================================================

__global__ __launch_bounds__(BLOCK_SIZE, 1)
void dsmem_cluster_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int elements_per_cluster
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cg::thread_block block = cg::this_thread_block();
    cg::cluster_group cluster = cg::this_cluster();
    
    const int cluster_id = blockIdx.x / CLUSTER_SIZE;
    const int cluster_rank = cluster.block_rank();
    const int tid = threadIdx.x;
    
    // Shared memory for reductions
    __shared__ float smem_reduce[32];  // For warp results
    __shared__ float smem_cluster[CLUSTER_SIZE];  // For cluster reduction
    
    // Global offset for this cluster
    const int cluster_offset = cluster_id * elements_per_cluster;
    const int block_offset = cluster_offset + cluster_rank * ELEMENTS_PER_BLOCK;
    
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
    
    // Block-level reduction
    float block_sum = block_reduce_sum(local_sum, smem_reduce);
    
    //========================================================================
    // STEP 2: Write block result to cluster-visible shared memory
    //========================================================================
    if (tid == 0) {
        smem_cluster[cluster_rank] = block_sum;
    }
    
    // Synchronize entire cluster
    cluster.sync();
    
    //========================================================================
    // STEP 3: Cluster leader reads all block results via DSMEM
    //========================================================================
    if (cluster_rank == 0) {
        float cluster_sum = smem_cluster[0];  // Own result
        
        // Read from peer CTAs via DSMEM
        #pragma unroll
        for (int peer = 1; peer < CLUSTER_SIZE; ++peer) {
            float* peer_smem = cluster.map_shared_rank(smem_cluster, peer);
            cluster_sum += peer_smem[peer];
        }
        
        // Single atomic add to global output
        if (tid == 0) {
            atomicAdd(&output[cluster_id], cluster_sum);
        }
    }
    
#else
    // Fallback for older architectures - standard block reduction
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
// Standard Reduction (Baseline) - Two-pass approach
//============================================================================

__global__ void standard_block_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ partial_sums,
    int N
) {
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
        partial_sums[blockIdx.x] = block_sum;
    }
}

__global__ void final_reduction_kernel(
    const float* __restrict__ partial_sums,
    float* __restrict__ output,
    int num_blocks,
    int blocks_per_cluster
) {
    const int cluster_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int start_block = cluster_id * blocks_per_cluster;
    
    __shared__ float smem_reduce[32];
    
    float local_sum = 0.0f;
    for (int i = tid; i < blocks_per_cluster; i += BLOCK_SIZE) {
        int block_idx = start_block + i;
        if (block_idx < num_blocks) {
            local_sum += partial_sums[block_idx];
        }
    }
    
    float block_sum = block_reduce_sum(local_sum, smem_reduce);
    
    if (tid == 0) {
        output[cluster_id] = block_sum;
    }
}

//============================================================================
// Benchmark
//============================================================================

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("DSMEM Cluster Reduction Example\n");
    printf("================================\n");
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    
    const bool has_clusters = prop.major >= 9;
    if (!has_clusters) {
        printf("\nNote: DSMEM requires SM 9.0+. Running fallback kernel.\n");
    }
    
    // Problem size
    const int N = 16 * 1024 * 1024;  // 16M elements
    const int elements_per_cluster = ELEMENTS_PER_BLOCK * CLUSTER_SIZE;
    const int num_clusters = (N + elements_per_cluster - 1) / elements_per_cluster;
    const int num_blocks = num_clusters * CLUSTER_SIZE;
    
    printf("\nProblem Size:\n");
    printf("  Elements: %d (%.1f MB)\n", N, N * sizeof(float) / 1e6);
    printf("  Clusters: %d\n", num_clusters);
    printf("  Blocks per cluster: %d\n", CLUSTER_SIZE);
    printf("  Total blocks: %d\n\n", num_blocks);
    
    // Allocate
    float *d_input, *d_output, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, num_clusters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, num_blocks * sizeof(float)));
    
    // Initialize with known pattern
    std::vector<float> h_input(N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;  // Sum should equal N
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int warmup = 5;
    const int iterations = 50;
    
    //========================================================================
    // Benchmark Standard Two-Pass Reduction
    //========================================================================
    printf("Standard Two-Pass Reduction:\n");
    
    CUDA_CHECK(cudaMemset(d_output, 0, num_clusters * sizeof(float)));
    
    for (int i = 0; i < warmup; ++i) {
        standard_block_reduction_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_partial, N);
        final_reduction_kernel<<<num_clusters, BLOCK_SIZE>>>(d_partial, d_output, num_blocks, CLUSTER_SIZE);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        standard_block_reduction_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_partial, N);
        final_reduction_kernel<<<num_clusters, BLOCK_SIZE>>>(d_partial, d_output, num_blocks, CLUSTER_SIZE);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_standard;
    CUDA_CHECK(cudaEventElapsedTime(&ms_standard, start, stop));
    float avg_standard = ms_standard / iterations;
    
    // Verify
    std::vector<float> h_output(num_clusters);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_clusters * sizeof(float), cudaMemcpyDeviceToHost));
    float total_standard = std::accumulate(h_output.begin(), h_output.end(), 0.0f);
    
    printf("  Time: %.3f ms\n", avg_standard);
    printf("  Sum: %.0f (expected: %d)\n", total_standard, N);
    
    //========================================================================
    // Benchmark DSMEM Cluster Reduction
    //========================================================================
    printf("\nDSMEM Cluster Reduction:\n");
    
    CUDA_CHECK(cudaMemset(d_output, 0, num_clusters * sizeof(float)));
    
    // Note: Full cluster launch would use cudaLaunchAttributeClusterDimension
    // This simplified version uses the fallback path
    for (int i = 0; i < warmup; ++i) {
        dsmem_cluster_reduction_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_input, d_output, N, elements_per_cluster);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        dsmem_cluster_reduction_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_input, d_output, N, elements_per_cluster);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_dsmem;
    CUDA_CHECK(cudaEventElapsedTime(&ms_dsmem, start, stop));
    float avg_dsmem = ms_dsmem / iterations;
    
    // Verify
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_clusters * sizeof(float), cudaMemcpyDeviceToHost));
    float total_dsmem = std::accumulate(h_output.begin(), h_output.end(), 0.0f);
    
    printf("  Time: %.3f ms\n", avg_dsmem);
    printf("  Sum: %.0f (expected: %d)\n", total_dsmem, N);
    
    //========================================================================
    // Summary
    //========================================================================
    printf("\nSummary:\n");
    float speedup = avg_standard / avg_dsmem;
    if (speedup > 1.0f) {
        printf("  DSMEM is %.2fx FASTER\n", speedup);
    } else {
        printf("  Standard is %.2fx faster (DSMEM requires cluster launch for full benefit)\n", 1.0f / speedup);
    }
    
    printf("\nNote: Full DSMEM benefit requires:\n");
    printf("  - cudaLaunchAttributeClusterDimension for cluster launch\n");
    printf("  - SM 9.0+ (Hopper/Blackwell)\n");
    printf("  - This example shows the pattern; full implementation needs cluster attributes\n");
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_partial));
    
    return 0;
}

