// optimized_cluster_group_no_dsmem.cu -- DSMEM-free optimized reduction using shared memory pipelines.

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#include "cluster_group_common.cuh"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _status = (call);                                           \
        if (_status != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s (%d) at %s:%d\n",                    \
                    cudaGetErrorString(_status), _status, __FILE__, __LINE__);  \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

constexpr int kWarpSize = 32;

__device__ float warp_reduce_sum(float val) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__device__ void block_reduce(T* shared, T val) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();
    if (warp == 0) {
        val = (lane < blockDim.x / kWarpSize) ? shared[lane] : static_cast<T>(0);
        val = warp_reduce_sum(val);
        if (lane == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();
}

__global__ void optimized_no_dsmem_kernel(const float* __restrict__ input,
                                          float* __restrict__ sum_out,
                                          float* __restrict__ sq_out,
                                          int chunk_elems,
                                          int total_chunks,
                                          int total_elements) {
    extern __shared__ float shared[];
    const int warps = blockDim.x / kWarpSize;
    float* shared_sum = shared;
    float* shared_sq = shared + warps;

    for (int chunk = blockIdx.x; chunk < total_chunks; chunk += gridDim.x) {
        const size_t base = static_cast<size_t>(chunk) * chunk_elems;
        if (base >= static_cast<size_t>(total_elements)) {
            continue;
        }
        float local_sum = 0.0f;
        float local_sq = 0.0f;

        for (int offset = threadIdx.x; offset < chunk_elems; offset += blockDim.x) {
            const size_t idx = base + static_cast<size_t>(offset);
            if (idx >= static_cast<size_t>(total_elements)) {
                break;
            }
            const float val = input[idx];
            local_sum += val;
            local_sq += val * val;
        }

        block_reduce(shared_sum, local_sum);
        block_reduce(shared_sq, local_sq);

        if (threadIdx.x == 0) {
            sum_out[chunk] = shared_sum[0];
            sq_out[chunk] = shared_sq[0];
        }
        __syncthreads();
    }
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    std::vector<float> h_input(kTotalElements);
    initialize_input(h_input);

    const int chunk_elems = kChunkElements;
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

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int blocks = std::min(chunks, prop.multiProcessorCount * 4);
    dim3 grid(blocks);
    dim3 block(kThreadsPerBlock);
    const size_t shared_bytes = 2 * (block.x / kWarpSize) * sizeof(float);

    optimized_no_dsmem_kernel<<<grid, block, shared_bytes>>>(
        d_input, d_sum, d_sq, chunk_elems, chunks, kTotalElements);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kIterations; ++i) {
        optimized_no_dsmem_kernel<<<grid, block, shared_bytes>>>(
            d_input, d_sum, d_sq, chunk_elems, chunks, kTotalElements);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const float avg_ms = elapsed_ms / static_cast<float>(kIterations);
    printf("Optimized (block-level reduction, no DSMEM): %.3f ms\n", avg_ms);
    printf("TIME_MS: %.6f\n", avg_ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_sq));
    return 0;
}
