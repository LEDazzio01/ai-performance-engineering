// optimized_graph_bandwidth.cu
// Optimized: CUDA Graph capture eliminates launch overhead
//
// Key concepts:
// - CUDA graphs capture entire kernel sequence once
// - Graph replay has minimal CPU overhead
// - Launch overhead amortized across all kernels in graph
// - Significant speedup for many small kernels

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
    constexpr int KERNELS_PER_GRAPH = 16;  // Same as baseline kernels per iter
    const size_t data_size_bytes = N * sizeof(float);
    
    printf("========================================\n");
    printf("OPTIMIZED: CUDA Graph Launch\n");
    printf("========================================\n");
    printf("Problem size: %d elements (%.2f KB)\n", N, data_size_bytes / 1024.0f);
    printf("Iterations: %d\n", ITERATIONS);
    printf("Kernels per graph: %d (captured once, replayed)\n", KERNELS_PER_GRAPH);
    printf("Total graph launches: %d\n\n", ITERATIONS);
    
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
    
    // Capture all 16 kernels into a single CUDA graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    // Capture same sequence as baseline - 16 kernels
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
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Measure - single graph launch runs all 16 kernels
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    printf("Running CUDA graph launch (optimized)...\n");
    
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < ITERATIONS; ++i) {
        // Single graph launch executes all 16 kernels
        // Launch overhead is paid once per graph, not 16 times
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    printf("\nResults:\n");
    printf("  Total time: %.3f ms\n", ms);
    printf("  Total graph launches: %d (each runs %d kernels)\n", ITERATIONS, KERNELS_PER_GRAPH);
    printf("  Avg time per graph: %.4f ms\n", ms / ITERATIONS);
    printf("  Effective kernels executed: %d\n", ITERATIONS * KERNELS_PER_GRAPH);
    
    printf("\n========================================\n");
    printf("Optimization Benefits:\n");
    printf("  - Graph captures %d kernels once\n", KERNELS_PER_GRAPH);
    printf("  - Replay launches all kernels together\n");
    printf("  - 16x fewer CPU->GPU launch commands\n");
    printf("  - Ideal for repetitive small-kernel workloads\n");
    printf("========================================\n");
    
    printf("\nTIME_MS: %.6f\n", ms);
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_tmp));
    
    return 0;
}
