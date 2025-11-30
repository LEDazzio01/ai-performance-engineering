// Baseline loop-unrolling binary: redundant accumulation with limited ILP.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "loop_unrolling_common.cuh"

using namespace ch08;

int main() {
    const int rows = 1 << 15;  // 32K rows
    const size_t input_elements = static_cast<size_t>(rows) * kElementsPerRow;
    const size_t input_bytes = input_elements * sizeof(float);
    const size_t weight_bytes = kWeightPeriod * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(rows) * sizeof(float);

    std::vector<float> h_inputs(input_elements);
    std::vector<float> h_weights(kWeightPeriod);
    std::vector<float> h_output(rows);

    for (size_t i = 0; i < input_elements; ++i) {
        h_inputs[i] = static_cast<float>((i % 1024) - 512) / 512.0f;
    }
    for (int i = 0; i < kWeightPeriod; ++i) {
        h_weights[i] = 0.5f + 0.1f * static_cast<float>(i);
    }

    float* d_inputs = nullptr;
    float* d_weights = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_inputs, input_bytes);
    cudaMalloc(&d_weights, weight_bytes);
    cudaMalloc(&d_output, output_bytes);

    cudaMemcpy(d_inputs, h_inputs.data(), input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights.data(), weight_bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        launch_loop_unrolling_baseline(d_inputs, d_weights, d_output, rows, nullptr);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    const float avg_ms = total_ms / static_cast<float>(iterations);
    std::cout << "Original loop (baseline): " << avg_ms << " ms\n";

    cudaMemcpy(h_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_inputs);
    cudaFree(d_weights);
    cudaFree(d_output);

    return 0;
}
