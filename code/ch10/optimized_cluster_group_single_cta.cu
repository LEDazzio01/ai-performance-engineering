// optimized_cluster_group_single_cta.cu -- single-CTA reduction fallback without thread-block clusters.

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

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void optimized_single_cta_kernel(const float* __restrict__ in,
                                            float* __restrict__ sum_out,
                                            float* __restrict__ sq_out,
                                            int chunk_elems,
                                            int total_elements) {
    const int chunk_id = blockIdx.x;
    const size_t chunk_base = static_cast<size_t>(chunk_id) * chunk_elems;
    if (chunk_base >= static_cast<size_t>(total_elements)) {
        return;
    }

    const int chunk_limit = min(chunk_elems, total_elements - static_cast<int>(chunk_base));
    const float* chunk_ptr = in + chunk_base;

    float sum = 0.0f;
    float sq_sum = 0.0f;

    const int thread_stride_vec = blockDim.x * 4;
    const int vec_limit = (chunk_limit / 4) * 4;
    const float4* chunk_ptr4 = reinterpret_cast<const float4*>(chunk_ptr);

    for (int offset = threadIdx.x * 4; offset < vec_limit; offset += thread_stride_vec) {
        const float4 v = chunk_ptr4[offset / 4];
        sum += v.x + v.y + v.z + v.w;
        sq_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    for (int offset = vec_limit + threadIdx.x; offset < chunk_limit; offset += blockDim.x) {
        const float v = chunk_ptr[offset];
        sum += v;
        sq_sum += v * v;
    }

    sum = warp_reduce_sum(sum);
    sq_sum = warp_reduce_sum(sq_sum);

    const int warp_id = threadIdx.x / warpSize;
    const int lane = threadIdx.x & (warpSize - 1);
    const int warp_count = blockDim.x / warpSize;

    extern __shared__ float shared[];
    float* warp_partial_sum = shared;
    float* warp_partial_sq = shared + warp_count;

    if (lane == 0) {
        warp_partial_sum[warp_id] = sum;
        warp_partial_sq[warp_id] = sq_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (lane < warp_count) ? warp_partial_sum[lane] : 0.0f;
        float block_sq_sum = (lane < warp_count) ? warp_partial_sq[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        block_sq_sum = warp_reduce_sum(block_sq_sum);
        if (lane == 0) {
            sum_out[chunk_id] = block_sum;
            sq_out[chunk_id] = block_sq_sum;
        }
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
    CUDA_CHECK(cudaMemset(d_sum, 0, result_bytes));
    CUDA_CHECK(cudaMemset(d_sq, 0, result_bytes));

    dim3 block(kThreadsPerBlock);
    dim3 grid(chunks);
    constexpr int warp_count = kThreadsPerBlock / 32;
    const size_t shared_bytes = sizeof(float) * 2 * warp_count;

    // Warmup
    optimized_single_cta_kernel<<<grid, block, shared_bytes>>>(
        d_input, d_sum, d_sq, chunk_elems, kTotalElements);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kIterations; ++i) {
        CUDA_CHECK(cudaMemsetAsync(d_sum, 0, result_bytes));
        CUDA_CHECK(cudaMemsetAsync(d_sq, 0, result_bytes));
        optimized_single_cta_kernel<<<grid, block, shared_bytes>>>(
            d_input, d_sum, d_sq, chunk_elems, kTotalElements);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const float avg_ms = elapsed_ms / static_cast<float>(kIterations);
    printf("Optimized single-CTA reduction: %.3f ms\n", avg_ms);
    printf("TIME_MS: %.6f\n", avg_ms);

    CUDA_CHECK(cudaMemset(d_sum, 0, result_bytes));
    CUDA_CHECK(cudaMemset(d_sq, 0, result_bytes));
    optimized_single_cta_kernel<<<grid, block, shared_bytes>>>(
        d_input, d_sum, d_sq, chunk_elems, kTotalElements);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_sum(chunks, 0.0f);
    std::vector<float> h_squares(chunks, 0.0f);
    CUDA_CHECK(cudaMemcpy(h_sum.data(), d_sum, result_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_squares.data(), d_sq, result_bytes, cudaMemcpyDeviceToHost));

    std::vector<float> ref_sum;
    std::vector<float> ref_squares;
    compute_reference(h_input, ref_sum, ref_squares);

    auto max_diff = [](const std::vector<float>& a, const std::vector<float>& b) {
        float diff = 0.0f;
        const std::size_t limit = std::min(a.size(), b.size());
        for (std::size_t i = 0; i < limit; ++i) {
            diff = std::max(diff, std::abs(a[i] - b[i]));
        }
        return diff;
    };

    printf("Verification (optimized single CTA): max |sum diff|=%.6f, |sq diff|=%.6f\n",
           max_diff(h_sum, ref_sum),
           max_diff(h_squares, ref_squares));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_sq));
    return 0;
}
