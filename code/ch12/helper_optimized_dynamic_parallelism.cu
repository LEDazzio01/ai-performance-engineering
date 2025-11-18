// optimized_dynamic_parallelism.cu -- device-launched segmented work (optimized).

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "dynamic_segments_common.cuh"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t status = (call);                                            \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                         __FILE__, __LINE__, cudaGetErrorString(status));       \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

__global__ void child_scale_kernel(float* data, int length, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) {
        return;
    }
    float v = data[idx];
    v = v * scale + 0.1f;
    data[idx] = tanhf(v);
}

__global__ void parent_launcher(float* data,
                                const Segment* segments,
                                int num_segments) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = tid; idx < num_segments; idx += stride) {
        Segment seg = segments[idx];
        constexpr int threads = 128;
        int blocks = (seg.length + threads - 1) / threads;
        child_scale_kernel<<<blocks, threads>>>(data + seg.offset, seg.length, seg.scale);
    }
}

int main() {
    constexpr int kElements = 1 << 22;
    constexpr int kGrid = 128;
    constexpr int kBlock = 128;
    constexpr int kWarmup = 1;
    constexpr int kIters = 10;

    std::vector<float> h_data(kElements);
    for (int i = 0; i < kElements; ++i) {
        h_data[i] = std::cos(0.0013f * static_cast<float>(i));
    }
    const std::vector<Segment> segments = build_segments(kElements);

    float* d_data = nullptr;
    float* d_seed = nullptr;
    Segment* d_segments = nullptr;
    const size_t bytes = h_data.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_seed, bytes));
    CUDA_CHECK(cudaMalloc(&d_segments, segments.size() * sizeof(Segment)));
    CUDA_CHECK(cudaMemcpy(d_segments, segments.data(), segments.size() * sizeof(Segment), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_seed, h_data.data(), bytes, cudaMemcpyHostToDevice));

    auto reset_input = [&]() {
        CUDA_CHECK(cudaMemcpy(d_data, d_seed, bytes, cudaMemcpyDeviceToDevice));
    };

    reset_input();
    for (int i = 0; i < kWarmup; ++i) {
        parent_launcher<<<kGrid, kBlock>>>(d_data, d_segments, static_cast<int>(segments.size()));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < kIters; ++iter) {
        reset_input();
        parent_launcher<<<kGrid, kBlock>>>(d_data, d_segments, static_cast<int>(segments.size()));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::printf("Dynamic optimized (device launches): %.3f ms\n", elapsed_ms / kIters);

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, h_data.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double checksum = 0.0;
    for (float v : h_data) {
        checksum += static_cast<double>(v);
    }
    std::printf("Optimized checksum: %.6e\n", checksum);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_seed));
    CUDA_CHECK(cudaFree(d_segments));
    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
