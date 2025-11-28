// baseline_cuda_graphs_conditional_enhanced.cu -- Standard approach (baseline).

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                              \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

constexpr int N = 1 << 20;

__global__ void expensive_kernel(float* data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            val = sqrtf(val * val + scale) * 0.99f;
        }
        data[idx] = val;
    }
}

__global__ void cheap_kernel(float* data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

__global__ void predicate_kernel(int* condition, float* data, int n, float threshold) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *condition = (data[0] > threshold) ? 1 : 0;
    }
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("=== Baseline Conditional Graphs Enhanced (Standard Dynamic Dispatch) ===\n");
    std::printf("Architecture: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    
    size_t bytes = N * sizeof(float);
    float *d_data = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    
    std::vector<float> h_data(N);
    for (int i = 0; i < N; ++i) {
        h_data[i] = 1.0f + (i % 100) * 0.01f;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    constexpr int ITERS = 1000;
    constexpr float THRESHOLD = 1.5f;
    
    int *d_condition = nullptr;
    CUDA_CHECK(cudaMalloc(&d_condition, sizeof(int)));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Baseline: host-side conditional dispatch with D2H synchronization
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        predicate_kernel<<<1, 1>>>(d_condition, d_data, N, THRESHOLD);
        
        int h_condition = 0;
        CUDA_CHECK(cudaMemcpy(&h_condition, d_condition, sizeof(int), cudaMemcpyDeviceToHost));
        
        if (h_condition) {
            expensive_kernel<<<grid, block>>>(d_data, N, 1.01f);
        } else {
            cheap_kernel<<<grid, block>>>(d_data, N, 0.99f);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    std::printf("Baseline (individual kernel launches): %.2f ms (%.3f ms/iter)\n", ms, ms / ITERS);
    std::printf("Note: Comparing kernel launch overhead vs CUDA graph replay\n");
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_condition));
    CUDA_CHECK(cudaFree(d_data));
    
    return 0;
}

