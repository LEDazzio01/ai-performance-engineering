// optimized_gemm_batched.cu - Batched GEMM (optimized with cublasSgemmBatched)
// Reduces kernel launch overhead by batching multiple GEMMs into one call

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

int main() {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    
    int m = 32;
    int n = 256;
    int k = 256;
    int batch_count = 40;
    
    // Allocate arrays of pointers for batched GEMM
    std::vector<float*> h_A(batch_count), h_B(batch_count), h_C(batch_count);
    
    for (int i = 0; i < batch_count; ++i) {
        CUDA_CHECK(cudaMalloc(&h_A[i], m * k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_B[i], k * n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_C[i], m * n * sizeof(float)));
        
        fill_matrix(h_A[i], m * k, 1.0f);
        fill_matrix(h_B[i], k * n, 1.0f);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
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
    
    float time_batched = ms / 100.0f;
    float tflops_batched = (2.0f * m * n * k * batch_count) / (time_batched * 1e9);
    
    std::printf("=== Optimized: Batched GEMM (cublasSgemmBatched) ===\n");
    std::printf("Matrix dimensions: M=%d, N=%d, K=%d, Batch=%d\n", m, n, k, batch_count);
    std::printf("Time: %.3f ms\n", time_batched);
    std::printf("Performance: %.2f TFLOPS\n", tflops_batched);
    std::printf("Kernel launches: 1 per iteration (batched)\n");
    
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
    CUBLAS_CHECK(cublasDestroy(handle));
    
    return 0;
}

