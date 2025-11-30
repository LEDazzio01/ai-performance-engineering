// baseline_gemm.cu - Individual GEMM calls (baseline, unoptimized)
// Demonstrates performance issue: many separate kernel launches

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
                    cudaGetErrorString(status));                             \
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

struct alignas(32) Float8 {
    float v[8];
};
static_assert(sizeof(Float8) == 32, "Float8 must pack eight floats");
static_assert(alignof(Float8) == 32, "Float8 must be 32-byte aligned");

__global__ void fill_matrix_kernel(float* data, int elements, float value) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_elems = elements / 8;
    Float8 vec_value;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        vec_value.v[i] = value;
    }
    Float8* vec_ptr = reinterpret_cast<Float8*>(data);
    for (int idx = thread_id; idx < vec_elems; idx += stride) {
        vec_ptr[idx] = vec_value;
    }
    const int tail_start = vec_elems * 8;
    for (int idx = tail_start + thread_id; idx < elements; idx += stride) {
        data[idx] = value;
    }
}

void fill_matrix(float* data, int elements, float value) {
    if (elements <= 0) {
        return;
    }
    const int threads = 256;
    int blocks = (elements + threads * 8 - 1) / (threads * 8);
    if (blocks <= 0) {
        blocks = 1;
    }
    fill_matrix_kernel<<<blocks, threads>>>(data, elements, value);
    CUDA_CHECK(cudaGetLastError());
}

// Benchmark individual GEMM calls (simulating original PyTorch behavior)
int main() {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    
    // Simulate the workload from profiling: 40 GEMMs with typical NN dimensions
    int m = 32;   // batch size
    int n = 256;  // output features
    int k = 256;  // input features
    int batch_count = 40;
    
    std::vector<float*> d_A(batch_count), d_B(batch_count), d_C(batch_count);
    
    // Allocate matrices for each GEMM
    for (int i = 0; i < batch_count; ++i) {
        CUDA_CHECK(cudaMalloc(&d_A[i], m * k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B[i], k * n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C[i], m * n * sizeof(float)));
        
        // Initialize with dummy data
        fill_matrix(d_A[i], m * k, 1.0f);
        fill_matrix(d_B[i], k * n, 1.0f);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
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
    
    float time_individual = ms / 100.0f;
    float tflops_individual = (2.0f * m * n * k * batch_count) / (time_individual * 1e9);
    
    std::printf("=== Baseline: Individual GEMM Calls ===\n");
    std::printf("Matrix dimensions: M=%d, N=%d, K=%d, Batch=%d\n", m, n, k, batch_count);
    std::printf("Time: %.3f ms\n", time_individual);
    std::printf("Performance: %.2f TFLOPS\n", tflops_individual);
    std::printf("Kernel launches: %d per iteration\n", batch_count);
    
    // Cleanup
    for (int i = 0; i < batch_count; ++i) {
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    
    return 0;
}
