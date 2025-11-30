// baseline_memory_transfer.cu - Traditional PCIe-based CPU-GPU transfers (baseline)
// Demonstrates standard PCIe transfers without Grace-Blackwell optimizations
// Compile: nvcc -O3 -std=c++17 -arch=sm_121 baseline_memory_transfer.cu -o baseline_memory_transfer_sm121

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>

#define CUDA_CHECK(call) \
  do { \
    cudaError_t status = (call); \
    if (status != cudaSuccess) { \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(status)); \
      std::exit(EXIT_FAILURE); \
    } \
  } while (0)

// Traditional kernel: Process data in GPU memory (requires explicit H2D/D2H copies)
__global__ void traditional_process_kernel(
    const float* __restrict__ gpu_input,
    float* __restrict__ gpu_output,
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = gpu_input[idx];
        
        // Compute
        float result = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            result += sqrtf(val * val + float(i)) * 0.125f;
        }
        
        gpu_output[idx] = result;
    }
}

int main() {
    const int N = 100 * 1024 * 1024;  // 100M elements
    const int iterations = 100;
    
    printf("=== Baseline: Traditional PCIe-based Transfers ===\n");
    printf("Array size: %d elements (%.1f MB)\n", N, N * sizeof(float) / 1e6);
    printf("Architecture: Standard PCIe (not Grace-Blackwell optimized)\n\n");
    
    // Host memory (CPU)
    std::vector<float> h_input(N);
    std::vector<float> h_output(N);
    
    // Initialize input
    for (int i = 0; i < N; i++) {
        h_input[i] = float(i % 1000) / 1000.0f;
    }
    
    // Device memory (GPU)
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));
    
    // Warmup
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    traditional_process_kernel<<<grid, block>>>(d_input, d_output, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark: Traditional approach with explicit copies
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; iter++) {
        // Explicit H2D copy (PCIe bottleneck)
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        
        // Kernel execution
        traditional_process_kernel<<<grid, block>>>(d_input, d_output, N);
        
        // Explicit D2H copy (PCIe bottleneck)
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / iterations;
    double bandwidth_gbs = (N * sizeof(float) * 2 / 1e9) / (avg_ms / 1000.0);  // 2x for H2D+D2H
    
    printf("Average time per iteration: %.3f ms\n", avg_ms);
    printf("Bandwidth: %.1f GB/s (PCIe-limited)\n", bandwidth_gbs);
    printf("Status: PCIe 5.0 bottleneck (~128 GB/s theoretical max)\n");
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return 0;
}

