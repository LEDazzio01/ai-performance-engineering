// baseline_cluster_group.cu -- Naive DSMEM-less implementation using per-element atomics.

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

__global__ void baseline_atomic_kernel(const float* __restrict__ in,
                                       float* __restrict__ sum_out,
                                       float* __restrict__ sq_out,
                                       int chunk_elems,
                                       int total_elements) {
    const int chunk_id = blockIdx.x;
    const size_t chunk_base = static_cast<size_t>(chunk_id) * chunk_elems;
    if (chunk_base >= static_cast<size_t>(total_elements)) {
        return;
    }

    for (int offset = threadIdx.x; offset < chunk_elems; offset += blockDim.x) {
        const size_t idx = chunk_base + static_cast<size_t>(offset);
        if (idx >= static_cast<size_t>(total_elements)) {
            break;
        }
        const float value = in[idx];
        atomicAdd(&sum_out[chunk_id], value);
        atomicAdd(&sq_out[chunk_id], value * value);
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

    // Warmup
    baseline_atomic_kernel<<<grid, block>>>(d_input, d_sum, d_sq, chunk_elems, kTotalElements);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kIterations; ++i) {
        CUDA_CHECK(cudaMemsetAsync(d_sum, 0, result_bytes));
        CUDA_CHECK(cudaMemsetAsync(d_sq, 0, result_bytes));
        baseline_atomic_kernel<<<grid, block>>>(d_input, d_sum, d_sq, chunk_elems, kTotalElements);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const float avg_ms = elapsed_ms / static_cast<float>(kIterations);
    printf("Baseline (per-element atomics): %.3f ms\n", avg_ms);
    printf("TIME_MS: %.6f\n", avg_ms);

    // Single run for verification
    CUDA_CHECK(cudaMemset(d_sum, 0, result_bytes));
    CUDA_CHECK(cudaMemset(d_sq, 0, result_bytes));
    baseline_atomic_kernel<<<grid, block>>>(d_input, d_sum, d_sq, chunk_elems, kTotalElements);
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

    printf("Verification (baseline): max |sum diff|=%.6f, |sq diff|=%.6f\n",
           max_diff(h_sum, ref_sum),
           max_diff(h_squares, ref_squares));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_sq));
    return 0;
}
