// kernel_fusion_cuda_demo.cu - Kernel fusion + CUDA graphs demo
// Demonstrates multiple kernel launches, CUDA graph capture, and a fused kernel.
//
// Key concepts:
// - Separate kernels: Multiple kernel launches with intermediate memory traffic
// - Fused kernel: Single kernel that combines operations, reducing memory traffic
// - CUDA Graphs: Capture and replay fused kernel sequences efficiently
// - Bandwidth optimization: Fused kernels reduce global memory round trips

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include "../core/common/headers/profiling_helpers.cuh"

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(status));                               \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                        \
  } while (0)

//------------------------------------------------------
// Separate kernels (baseline): Multiple kernel launches
// Each kernel reads from global memory, computes, writes back
// This causes multiple round trips to global memory
__global__ void kernel_add(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + 1.0f;
    }
}

__global__ void kernel_multiply(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

__global__ void kernel_sqrt(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrtf(data[idx]);
    }
}

//------------------------------------------------------
// Fused kernel (optimized): Single kernel combining all operations
// Reads once, performs all operations, writes once
// Reduces global memory traffic by eliminating intermediate writes
__global__ void kernel_fused(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Fused: add + multiply + sqrt in one pass
        float val = data[idx];
        val = val + 1.0f;      // Op 1
        val = val * 2.0f;      // Op 2 (uses register, not global memory)
        val = sqrtf(val);      // Op 3 (uses register, not global memory)
        data[idx] = val;       // Single write to global memory
    }
}

//------------------------------------------------------
// Measure bandwidth of a kernel sequence
float measure_bandwidth(
    void (*kernel)(float*, int),
    float* d_data,
    int n,
    int iterations,
    cudaStream_t stream,
    const char* name
) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        kernel<<<grid, block, 0, stream>>>(d_data, n);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Measure
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start, stream));
    {
        PROFILE_KERNEL_LAUNCH(name);
        for (int i = 0; i < iterations; ++i) {
            kernel<<<grid, block, 0, stream>>>(d_data, n);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // Calculate bandwidth: 2 * size (read + write) / time
    float size_gb = (n * sizeof(float) * iterations * 2) / (1024.0f * 1024.0f * 1024.0f);
    float bandwidth_gbs = size_gb / (ms / 1000.0f);
    
    return bandwidth_gbs;
}

int main() {
    constexpr int N = 10'000'000;  // 10M elements (~40 MB)
    constexpr int ITERATIONS = 100;
    
    printf("========================================\n");
    printf("Kernel Fusion with CUDA Graphs\n");
    printf("========================================\n");
    printf("Problem size: %d elements (%.2f MB)\n", N, N * sizeof(float) / 1024.0f / 1024.0f);
    printf("Iterations: %d\n\n", ITERATIONS);
    
    // Allocate host and device memory
    std::vector<float> h_data(N);
    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
    }
    
    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    
    // Copy initial data
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));
    
    //------------------------------------------------------
    // Test 1: Separate kernels (baseline)
    printf("1. Separate kernels (baseline):\n");
    printf("   - kernel_add (read, compute, write)\n");
    printf("   - kernel_multiply (read, compute, write)\n");
    printf("   - kernel_sqrt (read, compute, write)\n");
    printf("   Total: 3 global memory round trips per iteration\n");
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        kernel_add<<<grid, block, 0, stream>>>(d_data, N);
        kernel_multiply<<<grid, block, 0, stream>>>(d_data, N);
        kernel_sqrt<<<grid, block, 0, stream>>>(d_data, N);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Measure
    CUDA_CHECK(cudaEventRecord(start, stream));
    {
        PROFILE_KERNEL_LAUNCH("separate_kernels");
        for (int i = 0; i < ITERATIONS; ++i) {
            kernel_add<<<grid, block, 0, stream>>>(d_data, N);
            kernel_multiply<<<grid, block, 0, stream>>>(d_data, N);
            kernel_sqrt<<<grid, block, 0, stream>>>(d_data, N);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    float separate_ms;
    CUDA_CHECK(cudaEventElapsedTime(&separate_ms, start, stop));
    
    // Reset data
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));
    
    //------------------------------------------------------
    // Test 2: CUDA Graph with separate kernels
    printf("\n2. CUDA Graph with separate kernels:\n");
    printf("   - Captures kernel sequence\n");
    printf("   - Reduces launch overhead\n");
    printf("   - Still has 3 global memory round trips\n");
    
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    kernel_add<<<grid, block, 0, stream>>>(d_data, N);
    kernel_multiply<<<grid, block, 0, stream>>>(d_data, N);
    kernel_sqrt<<<grid, block, 0, stream>>>(d_data, N);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Measure
    CUDA_CHECK(cudaEventRecord(start, stream));
    {
        PROFILE_KERNEL_LAUNCH("graph_separate_kernels");
        for (int i = 0; i < ITERATIONS; ++i) {
            CUDA_CHECK(cudaGraphLaunch(exec, stream));
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    float graph_separate_ms;
    CUDA_CHECK(cudaEventElapsedTime(&graph_separate_ms, start, stop));
    
    // Reset data
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));
    
    //------------------------------------------------------
    // Test 3: Fused kernel (optimized)
    printf("\n3. Fused kernel (optimized):\n");
    printf("   - Single kernel combining all operations\n");
    printf("   - 1 global memory round trip per iteration\n");
    printf("   - Reduced memory traffic: 3x less than separate kernels\n");
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        kernel_fused<<<grid, block, 0, stream>>>(d_data, N);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Measure
    CUDA_CHECK(cudaEventRecord(start, stream));
    {
        PROFILE_KERNEL_LAUNCH("fused_kernel");
        for (int i = 0; i < ITERATIONS; ++i) {
            kernel_fused<<<grid, block, 0, stream>>>(d_data, N);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    float fused_ms;
    CUDA_CHECK(cudaEventElapsedTime(&fused_ms, start, stop));
    
    // Reset data
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));
    
    //------------------------------------------------------
    // Results
    printf("\n========================================\n");
    printf("Baseline Results:\n");
    printf("  Separate kernels:       %.3f ms\n", separate_ms);
    printf("  Graph (separate):       %.3f ms (%.2fx vs baseline)\n", 
           graph_separate_ms, separate_ms / graph_separate_ms);
    printf("  Fused kernel:           %.3f ms (%.2fx vs baseline)\n", 
           fused_ms, separate_ms / fused_ms);
    
    printf("\nKey insight:\n");
    printf("  - Separate kernels: 3 global memory round trips\n");
    printf("  - Fused kernel: 1 global memory round trip (3x less traffic)\n");
    printf("========================================\n");
    
    // Cleanup
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_data));
    
    return 0;
}
