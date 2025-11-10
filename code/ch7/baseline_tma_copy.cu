// baseline_tma_copy.cu -- Naive neighbor gather copy (baseline).

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kLookahead = 64;
constexpr int kElements = 1 << 25;  // 33,554,432 elements (~128 MB)
constexpr int kRedundantReads = 8;
constexpr bool kValidateOutput = false;

__host__ __device__ __forceinline__ float combine_values(float center, float near_val, float far_val) {
    // Blend three neighboring samples to emulate a small stencil.
    return fmaf(far_val, 0.125f, fmaf(near_val, 0.25f, center * 0.75f));
}

__global__ void naive_neighbor_copy_kernel(const float* __restrict__ src,
                                           float* __restrict__ dst,
                                           int n) {
    const int stride = blockDim.x * gridDim.x;
    const volatile float* vol_src = reinterpret_cast<const volatile float*>(src);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n) {
        float center = 0.0f;
#pragma unroll
        for (int repeat = 0; repeat < kRedundantReads; ++repeat) {
            center = vol_src[idx];
        }
        const int near_idx = (idx + 1 < n) ? (idx + 1) : (n - 1);
        const int far_idx = (idx + kLookahead < n) ? (idx + kLookahead) : (n - 1);
        float near_val = 0.0f;
        float far_val = 0.0f;
#pragma unroll
        for (int repeat = 0; repeat < kRedundantReads; ++repeat) {
            near_val = vol_src[near_idx];
            far_val = vol_src[far_idx];
        }
        dst[idx] = combine_values(center, near_val, far_val);
        idx += stride;
    }
}

float checksum(const std::vector<float>& data) {
    double sum = 0.0;
    for (float v : data) {
        sum += static_cast<double>(v);
    }
    return static_cast<float>(sum / static_cast<double>(data.size()));
}

}  // namespace

int main() {
    const size_t bytes = static_cast<size_t>(kElements) * sizeof(float);

    float *d_src = nullptr, *d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));

    float* h_input = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_input, bytes));
    for (int i = 0; i < kElements; ++i) {
        h_input[i] = static_cast<float>((i % 1024) - 512) / 128.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_src, h_input, bytes, cudaMemcpyHostToDevice));

    const int grid = 2 * static_cast<int>((kElements + kThreadsPerBlock - 1) / kThreadsPerBlock);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Warmup
    CUDA_CHECK(cudaMemcpyAsync(d_src, h_input, bytes, cudaMemcpyHostToDevice, stream));
    naive_neighbor_copy_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(d_src, d_dst, kElements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    constexpr int kIterations = 20;
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int iter = 0; iter < kIterations; ++iter) {
        CUDA_CHECK(cudaMemcpyAsync(d_src, h_input, bytes, cudaMemcpyHostToDevice, stream));
        naive_neighbor_copy_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(d_src, d_dst, kElements);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / kIterations;
    std::printf("Naive neighbor copy (baseline): %.3f ms\n", avg_ms);

    std::vector<float> h_output(kElements);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_dst, bytes, cudaMemcpyDeviceToHost));

    if (kValidateOutput) {
        std::vector<float> h_reference(kElements);
        for (int i = 0; i < kElements; ++i) {
            const int near_idx = (i + 1 < kElements) ? (i + 1) : (kElements - 1);
            const int far_idx = (i + kLookahead < kElements) ? (i + kLookahead) : (kElements - 1);
            h_reference[i] = combine_values(h_input[i], h_input[near_idx], h_input[far_idx]);
        }

        float max_error = 0.0f;
        for (int i = 0; i < kElements; ++i) {
            max_error = std::max(max_error, std::abs(h_reference[i] - h_output[i]));
        }
        std::printf("Output checksum: %.6f (max error %.6f)\n", checksum(h_output), max_error);
    } else {
        std::printf("Output checksum: %.6f\n", checksum(h_output));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(h_input));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    return 0;
}
