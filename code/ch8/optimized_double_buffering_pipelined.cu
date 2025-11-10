// Optimized double buffering binary.

#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#include "double_buffering_common.cuh"

using namespace ch8;

int main() {
    const int elements = kDoubleBufferBlock * kDoubleBufferTile * 8192;
    std::vector<float> host(elements);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : host) {
        v = dist(gen);
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, elements * sizeof(float));
    cudaMalloc(&d_output, (elements / kDoubleBufferTile) * sizeof(float));
    cudaMemcpy(d_input, host.data(), elements * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const size_t shared_bytes = kDoubleBufferBlock * 2 * sizeof(float);

    for (int i = 0; i < 5; ++i) {
        double_buffer_optimized_kernel<<<double_buffer_grid(elements), kDoubleBufferBlock, shared_bytes>>>(d_input, d_output, elements);
    }
    cudaDeviceSynchronize();

    const int iterations = 80;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        double_buffer_optimized_kernel<<<double_buffer_grid(elements), kDoubleBufferBlock, shared_bytes>>>(d_input, d_output, elements);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    std::cout << "Optimized double buffering: " << (total_ms / iterations) << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
