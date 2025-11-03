/**
 * Thread Block Clusters for Blackwell
 * ====================================
 * 
 * Blackwell SM 10.0 provides enhanced Thread Block Cluster support:
 * - Up to 8 CTAs per cluster (vs 4 on Hopper)
 * - Distributed Shared Memory (DSMEM) - 2 MB total
 * - Better scheduling for 148 SMs
 * - Cluster-wide synchronization
 * 
 * Key Benefits:
 * - Increased parallelism
 * - Shared data across thread blocks
 * - Reduced global memory traffic
 * 
 * Requirements: SM 10.0 (Blackwell), CUDA 13.0+
 * 
 * Compile:
 *   nvcc -O3 -std=c++17 -arch=sm_100 cluster_group_blackwell.cu -o cluster_group
 * 
 * Performance on B200:
 * - Up to 8 CTAs per cluster
 * - 2 MB distributed shared memory
 * - Better load balancing on 148 SMs
 */

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#include "../common/headers/arch_detection.cuh"

namespace cg = cooperative_groups;

// Kernel that sums values across blocks in a cluster using block-scoped shared arrays
__global__ void cluster_sum_kernel(const float *in, float *out, int elems_per_block) {
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block cta = cg::this_thread_block();

    extern __shared__ float sdata[]; // per-CTA shared buffer

    float sum = 0.0f;
    int base = blockIdx.x * elems_per_block;
    for (int i = threadIdx.x; i < elems_per_block; i += blockDim.x) {
        sum += in[base + i];
    }

    sdata[threadIdx.x] = sum;
    cta.sync();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        cta.sync();
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x] = sdata[0];
    }

    cluster.sync();

    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        int cluster_blocks = cluster.dim_blocks().x * cluster.dim_blocks().y * cluster.dim_blocks().z;
        int cluster_start = blockIdx.x - cluster.block_rank();

        float cluster_total = 0.0f;
        for (int b = 0; b < cluster_blocks && (cluster_start + b) < gridDim.x; ++b) {
            cluster_total += out[cluster_start + b];
        }
        out[cluster_start] = cluster_total;
    }
}

// ============================================================================
// Blackwell Thread Block Cluster Information
// ============================================================================

void print_cluster_info() {
    const auto& limits = cuda_arch::get_architecture_limits();

    printf("\n=== Thread Block Cluster Capabilities ===\n");

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    if (!limits.supports_clusters) {
        printf("⚠️  This device does not expose Thread Block Cluster support.\n");
        printf("    Clusters require Hopper (SM 9.0) or newer hardware.\n");
        return;
    }

    printf("Cluster capabilities:\n");
    printf("  Max CTAs per cluster : %d\n", limits.max_cluster_size);
    printf("  Max shared memory / block : %.1f KB\n", limits.max_shared_mem_per_block / 1024.0);
    printf("  Max threads / block : %d\n", limits.max_threads_per_block);

    if (prop.major >= 10) {
        printf("\nBlackwell notes:\n");
        printf("  • Up to 8 CTAs per cluster (2× Hopper)\n");
        printf("  • Distributed shared memory improves cross-CTA communication\n");
        printf("  • Designed for 148-SM parts with improved scheduling\n");
    } else if (prop.major == 9) {
        printf("\nHopper notes:\n");
        printf("  • Up to 4 CTAs per cluster\n");
        printf("  • Distributed shared memory available\n");
    }

    printf("\nProgramming tips:\n");
    printf("  1. Configure clusters with cudaLaunchAttributeClusterDimension.\n");
    printf("  2. Use cg::this_cluster() for cluster-wide synchronization.\n");
    printf("  3. Choose cluster sizes that divide the grid when possible.\n");
}

int main() {
    printf("=== Thread Block Clusters ===\n\n");

    const auto& limits = cuda_arch::get_architecture_limits();
    print_cluster_info();

    printf("\n=== Running Cluster Sum Example ===\n");

    const int cluster_size = limits.supports_clusters ? limits.max_cluster_size : 1;
    int num_blocks = std::max(cluster_size * 2, 8);
    int elems_per_block = 1 << 20; // 1M elements per block
    int threads = 256;
    size_t total_elems = size_t(num_blocks) * elems_per_block;
    size_t bytes = total_elems * sizeof(float);

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, num_blocks * sizeof(float));

    std::vector<float> h_in(total_elems, 1.0f);
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, num_blocks * sizeof(float));

    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(num_blocks, 1, 1);
    cfg.blockDim = dim3(threads, 1, 1);
    cfg.dynamicSmemBytes = threads * sizeof(float);

    cudaLaunchAttribute attr[1];
    if (limits.supports_clusters) {
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = cluster_size;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        cudaFuncSetAttribute(cluster_sum_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    } else {
        cfg.attrs = nullptr;
        cfg.numAttrs = 0;
    }

    cudaError_t err = cudaLaunchKernelEx(&cfg, cluster_sum_kernel, d_in, d_out, elems_per_block);
    if (err != cudaSuccess) {
        printf("Cluster launch not supported or failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        return 0;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        return -1;
    }

    std::vector<float> h_out(num_blocks, 0.0f);
    cudaMemcpy(h_out.data(), d_out, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    double expected_block = static_cast<double>(elems_per_block);
    double expected_cluster_total = expected_block * cluster_size;

    printf("cluster_group demo completed.\n");
    int sample_block = (num_blocks > 1) ? 1 : 0;
    printf(" - Partial sum (block %d): %.2f (expected %.2f)\n", sample_block, h_out[sample_block], expected_block);
    printf(" - Cluster total (cluster 0): %.2f (expected %.2f)\n", h_out[0], expected_cluster_total);

    cudaFree(d_in);
    cudaFree(d_out);
    
    printf("\n=== Summary ===\n");
    if (limits.supports_clusters) {
        printf("✓ Cluster launch succeeded with %d CTAs per cluster\n", cluster_size);
    } else {
        printf("ℹ️  Device lacks cluster support; executed as independent CTAs\n");
    }
    printf("\n");
    
    return 0;
}
