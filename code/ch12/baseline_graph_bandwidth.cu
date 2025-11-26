// baseline_graph_bandwidth.cu
// Baseline: Separate kernel launches WITHOUT CUDA graphs
//
// Key concepts:
// - Traditional separate kernel launches incur launch overhead
// - Many small kernels amplify the per-launch overhead
// - This is the baseline to compare against CUDA graph optimization

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../common/headers/profiling_helpers.cuh"

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(status));                              \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// Very small, fast kernels to emphasize launch overhead
__global__ void saxpy_kernel(float* y, const float* x, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

__global__ void scale_kernel(float* data, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

__global__ void add_kernel(float* c, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void copy_kernel(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

int main() {
    // Small data to make kernels fast (emphasize launch overhead)
    constexpr int N = 100'000;  // 100K elements (~400 KB)
    constexpr int ITERATIONS = 500;
    constexpr int KERNELS_PER_ITER = 16;  // Many small kernels per iteration
    const size_t data_size_bytes = N * sizeof(float);
    
    printf("========================================\n");
    printf("BASELINE: Separate Kernel Launches\n");
    printf("========================================\n");
    printf("Problem size: %d elements (%.2f KB)\n", N, data_size_bytes / 1024.0f);
    printf("Iterations: %d\n", ITERATIONS);
    printf("Kernels per iteration: %d (separate launches)\n", KERNELS_PER_ITER);
    printf("Total kernel launches: %d\n\n", ITERATIONS * KERNELS_PER_ITER);
    
    // Allocate memory
    std::vector<float> h_data(N);
    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i % 1000) / 1000.0f;
    }
    
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, data_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, data_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, data_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp, data_size_bytes));
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    
    CUDA_CHECK(cudaMemcpy(d_a, h_data.data(), data_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_data.data(), data_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, data_size_bytes));
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        saxpy_kernel<<<grid, block, 0, stream>>>(d_c, d_a, 1.0f, N);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Measure - launch many small kernels separately each iteration
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    printf("Running separate kernel launches (baseline)...\n");
    
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < ITERATIONS; ++i) {
        // 16 small kernel launches per iteration - accumulates launch overhead
        saxpy_kernel<<<grid, block, 0, stream>>>(d_c, d_a, 1.001f, N);
        scale_kernel<<<grid, block, 0, stream>>>(d_c, 0.999f, N);
        add_kernel<<<grid, block, 0, stream>>>(d_tmp, d_a, d_b, N);
        copy_kernel<<<grid, block, 0, stream>>>(d_c, d_tmp, N);
        
        saxpy_kernel<<<grid, block, 0, stream>>>(d_c, d_b, 1.001f, N);
        scale_kernel<<<grid, block, 0, stream>>>(d_c, 0.999f, N);
        add_kernel<<<grid, block, 0, stream>>>(d_tmp, d_b, d_c, N);
        copy_kernel<<<grid, block, 0, stream>>>(d_c, d_tmp, N);
        
        saxpy_kernel<<<grid, block, 0, stream>>>(d_c, d_a, 1.001f, N);
        scale_kernel<<<grid, block, 0, stream>>>(d_c, 0.999f, N);
        add_kernel<<<grid, block, 0, stream>>>(d_tmp, d_c, d_a, N);
        copy_kernel<<<grid, block, 0, stream>>>(d_c, d_tmp, N);
        
        saxpy_kernel<<<grid, block, 0, stream>>>(d_c, d_b, 1.001f, N);
        scale_kernel<<<grid, block, 0, stream>>>(d_c, 0.999f, N);
        add_kernel<<<grid, block, 0, stream>>>(d_tmp, d_a, d_c, N);
        copy_kernel<<<grid, block, 0, stream>>>(d_c, d_tmp, N);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    printf("\nResults:\n");
    printf("  Total time: %.3f ms\n", ms);
    printf("  Total kernel launches: %d\n", ITERATIONS * KERNELS_PER_ITER);
    printf("  Avg time per kernel: %.4f ms\n", ms / (ITERATIONS * KERNELS_PER_ITER));
    printf("  Avg time per iteration: %.4f ms\n", ms / ITERATIONS);
    
    printf("\n========================================\n");
    printf("Baseline Characteristics:\n");
    printf("  - %d kernel launches incur CPU->GPU overhead\n", ITERATIONS * KERNELS_PER_ITER);
    printf("  - Each launch has driver processing time\n");
    printf("  - No batching of launch commands\n");
    printf("========================================\n");
    
    printf("\nTIME_MS: %.6f\n", ms);
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_tmp));
    
    return 0;
}
