// batched_gemm_example.cu - Demonstrate batched GEMM optimization
// This addresses the 40 separate GEMM launches observed in profiling

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

#define CUBLAS_CHECK(call)                                                   \
  do {                                                                       \
    cublasStatus_t status = (call);                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                                   \
      std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,   \
                    status);                                                 \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// Benchmark individual GEMM calls (simulating original PyTorch behavior)
float benchmark_individual_gemms(cublasHandle_t handle, int m, int n, int k, int batch_count) {
    std::vector<float*> d_A(batch_count), d_B(batch_count), d_C(batch_count);
    
    // Allocate matrices for each GEMM
    for (int i = 0; i < batch_count; ++i) {
        CUDA_CHECK(cudaMalloc(&d_A[i], m * k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B[i], k * n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C[i], m * n * sizeof(float)));
        
        // Initialize with dummy data
        CUDA_CHECK(cudaMemset(d_A[i], 1, m * k * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_B[i], 1, k * n * sizeof(float)));
    }
    
    // Warmup
    const float alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < batch_count; ++i) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 n, m, k, &alpha,
                                 d_B[i], n, d_A[i], k,
                                 &beta, d_C[i], n));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < 100; ++iter) {
        for (int i = 0; i < batch_count; ++i) {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     n, m, k, &alpha,
                                     d_B[i], n, d_A[i], k,
                                     &beta, d_C[i], n));
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    // Cleanup
    for (int i = 0; i < batch_count; ++i) {
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return ms / 100.0f;  // Average per iteration
}

// Benchmark batched GEMM (optimized approach)
float benchmark_batched_gemm(cublasHandle_t handle, int m, int n, int k, int batch_count) {
    // Allocate arrays of pointers for batched GEMM
    std::vector<float*> h_A(batch_count), h_B(batch_count), h_C(batch_count);
    
    for (int i = 0; i < batch_count; ++i) {
        CUDA_CHECK(cudaMalloc(&h_A[i], m * k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_B[i], k * n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_C[i], m * n * sizeof(float)));
        
        CUDA_CHECK(cudaMemset(h_A[i], 1, m * k * sizeof(float)));
        CUDA_CHECK(cudaMemset(h_B[i], 1, k * n * sizeof(float)));
    }
    
    // Copy pointer arrays to device
    float **d_A_array, **d_B_array, **d_C_array;
    CUDA_CHECK(cudaMalloc(&d_A_array, batch_count * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_B_array, batch_count * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_C_array, batch_count * sizeof(float*)));
    
    CUDA_CHECK(cudaMemcpy(d_A_array, h_A.data(), batch_count * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_array, h_B.data(), batch_count * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_array, h_C.data(), batch_count * sizeof(float*), cudaMemcpyHostToDevice));
    
    // Warmup
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    n, m, k, &alpha,
                                    (const float**)d_B_array, n,
                                    (const float**)d_A_array, k,
                                    &beta, d_C_array, n, batch_count));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < 100; ++iter) {
        CUBLAS_CHECK(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                        n, m, k, &alpha,
                                        (const float**)d_B_array, n,
                                        (const float**)d_A_array, k,
                                        &beta, d_C_array, n, batch_count));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    // Cleanup
    for (int i = 0; i < batch_count; ++i) {
        CUDA_CHECK(cudaFree(h_A[i]));
        CUDA_CHECK(cudaFree(h_B[i]));
        CUDA_CHECK(cudaFree(h_C[i]));
    }
    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_B_array));
    CUDA_CHECK(cudaFree(d_C_array));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return ms / 100.0f;
}

// Benchmark strided batched GEMM (best for uniform matrices)
float benchmark_strided_batched_gemm(cublasHandle_t handle, int m, int n, int k, int batch_count) {
    // Allocate contiguous memory for all matrices
    float *d_A, *d_B, *d_C;
    size_t stride_A = m * k;
    size_t stride_B = k * n;
    size_t stride_C = m * n;
    
    CUDA_CHECK(cudaMalloc(&d_A, stride_A * batch_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, stride_B * batch_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, stride_C * batch_count * sizeof(float)));
    
    CUDA_CHECK(cudaMemset(d_A, 1, stride_A * batch_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_B, 1, stride_B * batch_count * sizeof(float)));
    
    // Warmup
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                           n, m, k, &alpha,
                                           d_B, n, stride_B,
                                           d_A, k, stride_A,
                                           &beta, d_C, n, stride_C,
                                           batch_count));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < 100; ++iter) {
        CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                               n, m, k, &alpha,
                                               d_B, n, stride_B,
                                               d_A, k, stride_A,
                                               &beta, d_C, n, stride_C,
                                               batch_count));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return ms / 100.0f;
}

int main() {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // Simulate the workload from profiling: 40 GEMMs with typical NN dimensions
    // From PyTorch profiling: aten::addmm with 100ms total, 40 calls
    int m = 32;   // batch size
    int n = 256;  // output features
    int k = 256;  // input features
    int batch_count = 40;
    
    std::printf("=== Batched GEMM Performance Comparison ===\n");
    std::printf("Matrix dimensions: M=%d, N=%d, K=%d, Batch=%d\n\n", m, n, k, batch_count);
    
    std::printf("Testing 3 approaches (100 iterations each):\n\n");
    
    // Test 1: Individual GEMMs (current PyTorch behavior)
    std::printf("1. Individual cublasSgemm calls (40 launches):\n");
    float time_individual = benchmark_individual_gemms(handle, m, n, k, batch_count);
    float tflops_individual = (2.0f * m * n * k * batch_count) / (time_individual * 1e9);
    std::printf("   Time: %.3f ms\n", time_individual);
    std::printf("   Performance: %.2f TFLOPS\n\n", tflops_individual);
    
    // Test 2: Batched GEMM
    std::printf("2. cublasSgemmBatched (1 launch, pointer array):\n");
    float time_batched = benchmark_batched_gemm(handle, m, n, k, batch_count);
    float tflops_batched = (2.0f * m * n * k * batch_count) / (time_batched * 1e9);
    std::printf("   Time: %.3f ms\n", time_batched);
    std::printf("   Performance: %.2f TFLOPS\n", tflops_batched);
    std::printf("   Speedup: %.2fx\n\n", time_individual / time_batched);
    
    // Test 3: Strided Batched GEMM (best for uniform matrices)
    std::printf("3. cublasSgemmStridedBatched (1 launch, contiguous):\n");
    float time_strided = benchmark_strided_batched_gemm(handle, m, n, k, batch_count);
    float tflops_strided = (2.0f * m * n * k * batch_count) / (time_strided * 1e9);
    std::printf("   Time: %.3f ms\n", time_strided);
    std::printf("   Performance: %.2f TFLOPS\n", tflops_strided);
    std::printf("   Speedup: %.2fx\n\n", time_individual / time_strided);
    
    std::printf("=== Summary ===\n");
    std::printf("Batched GEMM reduces kernel launch overhead by %.1fx\n", time_individual / time_batched);
    std::printf("Strided Batched GEMM is optimal for uniform batch operations\n");
    std::printf("\nRecommendation for PyTorch:\n");
    std::printf("- For training: Use torch.bmm() or torch.matmul() with batched tensors\n");
    std::printf("- This automatically uses cublasSgemmStridedBatched\n");
    std::printf("- Avoids 40 separate kernel launches observed in profiling\n");
    
    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}

