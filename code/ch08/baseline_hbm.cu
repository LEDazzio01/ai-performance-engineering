// Baseline HBM binary.

#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#include "hbm_common.cuh"

using namespace ch08;

int main() {
    const int rows = 4096;
    const int cols = 2048;
    const size_t row_bytes = static_cast<size_t>(rows) * cols * sizeof(float);
    const size_t col_bytes = static_cast<size_t>(cols) * rows * sizeof(float);

    std::vector<float> host_matrix(rows * cols);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& value : host_matrix) {
        value = dist(gen);
    }

    std::vector<float> host_col(cols * rows);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            host_col[c * rows + r] = host_matrix[r * cols + c];
        }
    }

    float* d_col = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_col, col_bytes);
    cudaMalloc(&d_output, rows * sizeof(float));
    cudaMemcpy(d_col, host_col.data(), col_bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 5; ++i) {
        launch_hbm_naive(d_col, d_output, rows, cols, 0);
    }
    cudaDeviceSynchronize();

    const int iterations = 30;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        launch_hbm_naive(d_col, d_output, rows, cols, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    std::cout << "Baseline HBM: " << (total_ms / iterations) << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_col);
    cudaFree(d_output);
    return 0;
}
