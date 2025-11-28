// optimized_uneven_partition.cu -- device work stealing for uneven partitions.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "uneven_partition_common.cuh"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t status = (call);                                            \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                         __FILE__, __LINE__, cudaGetErrorString(status));       \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

// Use device pointer instead of __device__ variable to avoid cudaMemcpyToSymbol overhead
__global__ void dynamic_partition_kernel(const float* in,
                                         float* out,
                                         const UnevenSegment* segments,
                                         int num_segments,
                                         int* next_segment) {
    __shared__ int shared_segment;

    while (true) {
        __syncthreads();
        if (threadIdx.x == 0) {
            shared_segment = atomicAdd(next_segment, 1);
        }
        __syncthreads();
        int seg_idx = shared_segment;
        if (seg_idx >= num_segments) {
            break;
        }
        UnevenSegment seg = segments[seg_idx];
        for (int idx = threadIdx.x; idx < seg.length; idx += blockDim.x) {
            const int global_idx = seg.offset + idx;
            float v = in[global_idx];
            out[global_idx] = v * v + 0.5f * v;
        }
    }
}

int main() {
    constexpr int elems = (1 << 20) + 153;
    constexpr int grid_blocks = 192;
    constexpr int block_threads = 256;
    constexpr int warmup = 1;
    constexpr int iters = 10;

    std::vector<float> h_in(elems);
    std::vector<float> h_out(elems, 0.0f);
    for (int i = 0; i < elems; ++i) {
        h_in[i] = std::cos(0.00045f * static_cast<float>(i));
    }
    const std::vector<UnevenSegment> segments = build_uneven_segments(elems);

    float *d_in = nullptr, *d_out = nullptr;
    UnevenSegment* d_segments = nullptr;
    int* d_next_segment = nullptr;  // Device counter to avoid cudaMemcpyToSymbol
    CUDA_CHECK(cudaMalloc(&d_in, elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_segments, segments.size() * sizeof(UnevenSegment)));
    CUDA_CHECK(cudaMalloc(&d_next_segment, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_segments, segments.data(), segments.size() * sizeof(UnevenSegment), cudaMemcpyHostToDevice));

    auto launch_dynamic = [&]() {
        // Use cudaMemsetAsync instead of cudaMemcpyToSymbol for better performance
        CUDA_CHECK(cudaMemsetAsync(d_next_segment, 0, sizeof(int)));
        dynamic_partition_kernel<<<grid_blocks, block_threads>>>(
            d_in, d_out, d_segments, static_cast<int>(segments.size()), d_next_segment);
    };

    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(cudaMemset(d_out, 0, elems * sizeof(float)));
        launch_dynamic();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start_evt, stop_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));

    CUDA_CHECK(cudaEventRecord(start_evt));
    for (int iter = 0; iter < iters; ++iter) {
        CUDA_CHECK(cudaMemset(d_out, 0, elems * sizeof(float)));
        launch_dynamic();
    }
    CUDA_CHECK(cudaEventRecord(stop_evt));
    CUDA_CHECK(cudaEventSynchronize(stop_evt));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
    std::printf("Uneven optimized (device work stealing): %.3f ms\n", elapsed_ms / iters);

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, elems * sizeof(float), cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    for (int i = 0; i < elems; ++i) {
        const double input = static_cast<double>(h_in[i]);
        const double expected = input * input + 0.5 * input;
        max_err = std::max(max_err, std::abs(static_cast<double>(h_out[i]) - expected));
    }
    std::printf("Optimized uneven max error: %.3e\n", max_err);

    CUDA_CHECK(cudaEventDestroy(start_evt));
    CUDA_CHECK(cudaEventDestroy(stop_evt));
    CUDA_CHECK(cudaFree(d_next_segment));
    CUDA_CHECK(cudaFree(d_segments));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
