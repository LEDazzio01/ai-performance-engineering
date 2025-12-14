// optimized_cluster_group.cu -- Cluster DSMEM demo with fail-fast probe.

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <vector>

#include "cluster_group_common.cuh"
#include "../core/common/headers/cuda_verify.cuh"

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _status = (call);                                           \
        if (_status != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s (%d) at %s:%d\n",                    \
                    cudaGetErrorString(_status), _status, __FILE__, __LINE__);  \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

// Main kernel WITHOUT __cluster_dims__ attribute - use runtime specification only.
// The attribute conflicts with cudaLaunchAttributeClusterDimension on B200/CUDA 13.
// Use static shared memory for DSMEM compatibility on B200 (dynamic extern fails).
__global__ void cluster_dual_kernel(const float* __restrict__ input,
                    float* __restrict__ chunk_sum,
                    float* __restrict__ chunk_sq,
                    int elems_per_block,
                    int total_elements) {
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block block = cg::this_thread_block();

    // Static shared memory required for DSMEM on Blackwell B200.
    __shared__ float tile[kElementsPerBlock];
    __shared__ float reductions[kThreadsPerBlock];

    const int chunk_id = blockIdx.x / kClusterBlocks;
    const int cluster_rank = cluster.block_rank();
    const size_t base = static_cast<size_t>(chunk_id) * elems_per_block;
    if (base >= static_cast<size_t>(total_elements)) {
        return;
    }
    const int remaining = total_elements - static_cast<int>(base);
    const int valid = remaining < elems_per_block ? remaining : elems_per_block;

    if (cluster_rank == 0) {
        for (int i = threadIdx.x; i < valid; i += blockDim.x) {
            tile[i] = input[base + i];
        }
    }

    cluster.sync();
    const float* source_tile = cluster.map_shared_rank(tile, 0);

    float local = 0.0f;
    for (int i = threadIdx.x; i < valid; i += blockDim.x) {
        const float value = source_tile[i];
        local += (cluster_rank == 0) ? value : value * value;
    }

    reductions[threadIdx.x] = local;
    block.sync();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reductions[threadIdx.x] += reductions[threadIdx.x + stride];
        }
        block.sync();
    }

    if (threadIdx.x == 0) {
        if (cluster_rank == 0) {
            chunk_sum[chunk_id] = reductions[0];
        } else {
            chunk_sq[chunk_id] = reductions[0];
        }
    }

    // Ensure rank 0 keeps DSMEM tile alive until all ranks finish reading it.
    cluster.sync();
}

// Probe kernel WITHOUT __cluster_dims__ attribute - use runtime specification only.
// The attribute conflicts with cudaLaunchAttributeClusterDimension on B200/CUDA 13.
__global__ void dsmem_probe_kernel(float* out) {
    cg::cluster_group cluster = cg::this_cluster();
    // Static shared memory required for DSMEM on Blackwell B200 (dynamic extern fails)
    __shared__ float buffer[32];
    unsigned int block_rank = cluster.block_rank();

    if (block_rank == 0 && threadIdx.x == 0) {
        buffer[0] = 123.0f;
    }
    cluster.sync();
    if (block_rank == 1 && threadIdx.x == 0) {
        float* remote = cluster.map_shared_rank(buffer, 0);
        out[0] = remote[0];
    }
    cluster.sync();  // Final sync for clean exit
}

struct ProbeResult {
    bool ok;
    const char* stage;
    cudaError_t error;
};

ProbeResult probe_dsmem_support() {
    float* d_out = nullptr;
    cudaError_t alloc_err = cudaMalloc(&d_out, sizeof(float));
    if (alloc_err != cudaSuccess) {
        return {false, "cudaMalloc", alloc_err};
    }

    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(2);
    cfg.blockDim = dim3(32);
    cfg.dynamicSmemBytes = 0;  // Using static shared memory for DSMEM compatibility

    // Explicitly set cluster dimensions (required for B200 DSMEM support)
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    void* args[] = {&d_out};
    cudaError_t err = cudaLaunchKernelExC(&cfg, (void*)dsmem_probe_kernel, args);
    if (err != cudaSuccess) {
        cudaFree(d_out);
        return {false, "cudaLaunchKernelExC", err};
    }

    err = cudaDeviceSynchronize();
    cudaFree(d_out);
    if (err != cudaSuccess) {
        return {false, "cudaDeviceSynchronize", err};
    }
    return {true, nullptr, cudaSuccess};
}

int main() {
#if !defined(__CUDACC_VER_MAJOR__) || (__CUDACC_VER_MAJOR__ < 13)
    fprintf(stderr, "SKIPPED: CUDA 13+ required for cluster DSMEM support.\n");
    return 3;
#endif

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int cluster_launch = 0;
#ifdef cudaDevAttrClusterLaunch
    CUDA_CHECK(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, 0));
#endif
    const bool supports_clusters = (cluster_launch > 0) || (prop.major >= 9);
    if (!supports_clusters) {
        fprintf(stderr,
                "SKIPPED: Thread block clusters unsupported on %s (SM %d.%d).\n",
                prop.name,
                prop.major,
                prop.minor);
        return 3;
    }

    ProbeResult probe = probe_dsmem_support();
    if (!probe.ok) {
        fprintf(stderr,
                "SKIPPED: Distributed shared memory unavailable on %s (SM %d.%d). Stage=%s error=%s\n",
                prop.name,
                prop.major,
                prop.minor,
                probe.stage,
                cudaGetErrorString(probe.error));
        fprintf(stderr,
                "Use optimized_cluster_group_dram_partial_cluster_sync_no_dsmem_sm%03d for a DSMEM-free demonstration on this system.\n",
                prop.major * 10 + prop.minor);
        return 3;
    }

    std::vector<float> h_input(kTotalElements);
    initialize_input(h_input);

    const int chunks = num_chunks();
    const size_t input_bytes = h_input.size() * sizeof(float);
    const size_t result_bytes = chunks * sizeof(float);

    float* d_input = nullptr;
    float* d_sum = nullptr;
    float* d_sq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_sum, result_bytes));
    CUDA_CHECK(cudaMalloc(&d_sq, result_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));

    dim3 block(kThreadsPerBlock);
    dim3 grid(chunks * kClusterBlocks);
    // Using static shared memory now, so dynamicSmemBytes = 0
    const size_t shared_bytes = 0;

    cudaLaunchAttribute attrs[1]{};
    int attr_count = 0;
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13)
    attrs[attr_count].id = cudaLaunchAttributeClusterDimension;
    attrs[attr_count].val.clusterDim.x = kClusterBlocks;
    attrs[attr_count].val.clusterDim.y = 1;
    attrs[attr_count].val.clusterDim.z = 1;
    ++attr_count;

    CUDA_CHECK(cudaFuncSetAttribute(
        cluster_dual_kernel,
        cudaFuncAttributeNonPortableClusterSizeAllowed,
        1));
#endif

    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = shared_bytes;
    config.stream = nullptr;
    config.attrs = attr_count ? attrs : nullptr;
    config.numAttrs = attr_count;

    int elems_per_block = kElementsPerBlock;
    int total_elements = kTotalElements;
    void* args[] = {&d_input, &d_sum, &d_sq, &elems_per_block, &total_elements};

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kIterations; ++i) {
        CUDA_CHECK(cudaLaunchKernelExC(&config, (void*)cluster_dual_kernel, args));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const float avg_ms = elapsed_ms / static_cast<float>(kIterations);
    printf("Cluster launch (DSMEM optimized): %.3f ms\n", avg_ms);
    printf("TIME_MS: %.6f\n", avg_ms);

    // Single run for verification
    CUDA_CHECK(cudaMemset(d_sum, 0, result_bytes));
    CUDA_CHECK(cudaMemset(d_sq, 0, result_bytes));
    CUDA_CHECK(cudaLaunchKernelExC(&config, (void*)cluster_dual_kernel, args));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_sum(chunks, 0.0f);
    std::vector<float> h_squares(chunks, 0.0f);
    CUDA_CHECK(cudaMemcpy(h_sum.data(), d_sum, result_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_squares.data(), d_sq, result_bytes, cudaMemcpyDeviceToHost));

    std::vector<float> ref_sum;
    std::vector<float> ref_squares;
    compute_reference(h_input, ref_sum, ref_squares, kElementsPerBlock);

    auto max_diff = [](const std::vector<float>& a, const std::vector<float>& b) {
        float diff = 0.0f;
        const std::size_t limit = std::min(a.size(), b.size());
        for (std::size_t i = 0; i < limit; ++i) {
            diff = std::max(diff, std::abs(a[i] - b[i]));
        }
        return diff;
    };

    printf("Verification (optimized): max |sum diff|=%.6f, |sq diff|=%.6f\n",
           max_diff(h_sum, ref_sum),
           max_diff(h_squares, ref_squares));

    double checksum = 0.0;
    for (int i = 0; i < chunks; ++i) {
        checksum += static_cast<double>(h_sum[i]) + static_cast<double>(h_squares[i]);
    }
    checksum /= static_cast<double>(chunks);
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_sq));
    return 0;
}
