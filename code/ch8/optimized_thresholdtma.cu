// Optimized threshold binary using CUDA pipeline/TMA staging (Blackwell only).

#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#include "blackwell_guard.cuh"
#include "threshold_common.cuh"
#include "threshold_tma_kernel.cuh"

using namespace ch8;

namespace {

bool ensure_blackwell(int& exit_code) {
    cudaDeviceProp props{};
    cudaError_t err = cudaSuccess;
    if (is_blackwell_device(&props, &err, true)) {
        return true;
    }

    if (err == cudaSuccess && props.major > 0) {
        std::cerr << "SKIPPED: threshold_tma requires SM 12.x (Blackwell), found SM "
                  << props.major << "." << props.minor << "\n";
    } else {
        std::cerr << "SKIPPED: threshold_tma requires Blackwell/GB GPUs ("
                  << cudaGetErrorString(err) << ")\n";
    }
    exit_code = 3;
    return false;
}

void check(cudaError_t err, const char* label) {
    if (err != cudaSuccess) {
        std::cerr << label << ": " << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main() {
    int skip_code = 0;
    if (!ensure_blackwell(skip_code)) {
        return skip_code;
    }

    const int count = 1 << 25;
    const float threshold = 0.25f;
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);

    std::vector<float> h_input(count);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < count; ++i) {
        h_input[i] = dist(gen);
    }

    float* d_input = nullptr;
    float* d_output = nullptr;
    check(cudaMalloc(&d_input, bytes), "cudaMalloc d_input");
    check(cudaMalloc(&d_output, bytes), "cudaMalloc d_output");
    check(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 5; ++i) {
        check(
            launch_threshold_tma_pipeline(d_input, d_output, threshold, count, 0),
            "Warmup launch");
    }
    cudaDeviceSynchronize();

    const int iterations = 50;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        check(
            launch_threshold_tma_pipeline(d_input, d_output, threshold, count, 0),
            "Benchmark launch");
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    const float avg_ms = total_ms / iterations;

    std::cout << "=== Optimized Threshold (TMA pipeline) ===\n";
    std::cout << "Elements: " << count << " (" << bytes / 1e6 << " MB)\n";
    std::cout << "Average kernel time: " << avg_ms << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
