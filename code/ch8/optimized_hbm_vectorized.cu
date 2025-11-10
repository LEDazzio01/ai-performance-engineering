// Optimized HBM binary with vectorized accesses.

#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#include "hbm_common.cuh"

using namespace ch8;

int main() {
    const int rows = 4096;
    const int cols = 2048;
    const size_t bytes = static_cast<size_t>(rows) * cols * sizeof(float);

    std::vector<float> host_matrix(rows * cols);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& value : host_matrix) {
        value = dist(gen);
    }

    float* d_row = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_row, bytes);
    cudaMalloc(&d_output, rows * sizeof(float));
    cudaMemcpy(d_row, host_matrix.data(), bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 5; ++i) {
        launch_hbm_vectorized(d_row, d_output, rows, cols, 0);
    }
    cudaDeviceSynchronize();

    const int iterations = 50;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        launch_hbm_vectorized(d_row, d_output, rows, cols, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    std::cout << "Optimized HBM: " << (total_ms / iterations) << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_row);
    cudaFree(d_output);
    return 0;
}

