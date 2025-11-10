// baseline_uneven_partition.cu -- host-driven uneven partitions (baseline).

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

__global__ void static_partition_kernel(const float* in,
                                        float* out,
                                        int elems,
                                        int start,
                                        int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    const int global_idx = start + idx;
    if (global_idx < elems) {
        float v = in[global_idx];
        out[global_idx] = v * v + 0.5f * v;
    }
}

int main() {
    constexpr int elems = (1 << 20) + 153;
    constexpr int warmup = 1;
    constexpr int iters = 10;

    std::vector<float> h_in(elems);
    std::vector<float> h_out(elems, 0.0f);
    for (int i = 0; i < elems; ++i) {
        h_in[i] = std::sin(0.0005f * static_cast<float>(i));
    }

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), elems * sizeof(float), cudaMemcpyHostToDevice));

    const std::vector<UnevenSegment> segments = build_uneven_segments(elems);

    auto run_static = [&](cudaStream_t stream) {
        for (const UnevenSegment& seg : segments) {
            int blocks = (seg.length + 255) / 256;
            static_partition_kernel<<<blocks, 256, 0, stream>>>(
                d_in,
                d_out,
                elems,
                seg.offset,
                seg.length);
        }
    };

    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(cudaMemset(d_out, 0, elems * sizeof(float)));
        run_static(nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start_evt, stop_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));

    CUDA_CHECK(cudaEventRecord(start_evt));
    for (int iter = 0; iter < iters; ++iter) {
        CUDA_CHECK(cudaMemset(d_out, 0, elems * sizeof(float)));
        run_static(nullptr);
    }
    CUDA_CHECK(cudaEventRecord(stop_evt));
    CUDA_CHECK(cudaEventSynchronize(stop_evt));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
    std::printf("Uneven baseline (static partitions): %.3f ms\n", elapsed_ms / iters);

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, elems * sizeof(float), cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    for (int i = 0; i < elems; ++i) {
        const double input = static_cast<double>(h_in[i]);
        const double expected = input * input + 0.5 * input;
        max_err = std::max(max_err, std::abs(static_cast<double>(h_out[i]) - expected));
    }
    std::printf("Baseline uneven max error: %.3e\n", max_err);

    CUDA_CHECK(cudaEventDestroy(start_evt));
    CUDA_CHECK(cudaEventDestroy(stop_evt));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
