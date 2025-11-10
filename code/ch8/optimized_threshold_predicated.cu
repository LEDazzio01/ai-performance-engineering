// Optimized predicated threshold binary.

#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "threshold_common.cuh"
#include "threshold_async_kernel.cuh"

using namespace ch8;

int main() {
    const int count = 1 << 26;
    const float threshold = 0.5f;
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);

    std::vector<float> h_input(count);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < count; ++i) {
        h_input[i] = dist(gen);
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    bool async_available = true;

    auto run_threshold = [&](const char* phase) {
        if (async_available) {
            cudaError_t launch_err = cudaSuccess;
            const auto status = launch_threshold_predicated_async(
                d_input,
                d_output,
                threshold,
                count,
                0,
                &launch_err);
            if (status == ThresholdAsyncLaunchResult::kSuccess) {
                return;
            }
            if (status == ThresholdAsyncLaunchResult::kFailed) {
                std::cerr << "Async threshold launch failed during " << phase
                          << ": " << cudaGetErrorString(launch_err) << "\n";
                std::exit(EXIT_FAILURE);
            }
            static bool warned_async_unavailable = false;
            if (!warned_async_unavailable) {
                std::cerr << "Async threshold kernel unavailable ("
                          << cudaGetErrorString(launch_err)
                          << "); falling back to predicated path.\n";
                warned_async_unavailable = true;
            }
            async_available = false;
        }
        launch_threshold_predicated(d_input, d_output, threshold, count, 0);
    };

    for (int i = 0; i < 5; ++i) {
        run_threshold("warmup");
    }
    cudaDeviceSynchronize();

    const int iterations = 50;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        run_threshold("benchmark");
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    const float avg_ms = total_ms / iterations;

    std::cout << "=== Optimized Threshold (predicated) ===\n";
    std::cout << "Elements: " << count << " (" << bytes / 1e6 << " MB)\n";
    std::cout << "Average kernel time: " << avg_ms << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
