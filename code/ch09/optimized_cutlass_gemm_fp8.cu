// optimized_cutlass_gemm_fp8.cu -- FP8 GEMM using cuBLASLt tensor cores
//
// This optimized version uses cuBLASLt with FP8 (E4M3) data types and
// tensor core acceleration for maximum throughput on Blackwell/Hopper.
//
// BOOK REFERENCE (Ch9): FP8 tensor cores provide ~2x throughput over FP16
// and ~4x over FP32 due to reduced precision and increased tensor core density.
//
// Expected speedup: 1.4-1.6x over naive FP8 implementation.

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <iostream>
#include <random>

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t status = (call);                                                 \
    if (status != cudaSuccess) {                                                 \
      std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " "           \
                << cudaGetErrorString(status) << std::endl;                      \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#define CUBLASLT_CHECK(call)                                                     \
  do {                                                                           \
    cublasStatus_t status = (call);                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "cuBLASLt error " << __FILE__ << ":" << __LINE__ << " "        \
                << status << std::endl;                                          \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

int main() {
    // Matrix dimensions - same as baseline for fair comparison
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int K = 4096;
    constexpr int kIterations = 10;
    constexpr int kBatchCount = 8;

    const size_t elements_A = static_cast<size_t>(M) * K;
    const size_t elements_B = static_cast<size_t>(K) * N;
    const size_t elements_C = static_cast<size_t>(M) * N;
    const size_t size_A_fp8 = elements_A * sizeof(__nv_fp8_e4m3) * kBatchCount;
    const size_t size_B_fp8 = elements_B * sizeof(__nv_fp8_e4m3) * kBatchCount;
    const size_t size_C_fp16 = elements_C * sizeof(__half) * kBatchCount;  // Output in FP16

    // Host allocation with pinned memory
    __nv_fp8_e4m3* h_A = nullptr;
    __nv_fp8_e4m3* h_B = nullptr;
    __half* h_C = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_A, size_A_fp8));
    CUDA_CHECK(cudaMallocHost(&h_B, size_B_fp8));
    CUDA_CHECK(cudaMallocHost(&h_C, size_C_fp16));

    // Initialize with random FP8 values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
    for (size_t i = 0; i < elements_A * kBatchCount; ++i) {
        h_A[i] = __nv_fp8_e4m3(dis(gen));
    }
    for (size_t i = 0; i < elements_B * kBatchCount; ++i) {
        h_B[i] = __nv_fp8_e4m3(dis(gen));
    }
    for (size_t i = 0; i < elements_C * kBatchCount; ++i) {
        h_C[i] = __half(0.0f);
    }

    // Device allocation
    __nv_fp8_e4m3 *d_A = nullptr, *d_B = nullptr;
    __half *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size_A_fp8));
    CUDA_CHECK(cudaMalloc(&d_B, size_B_fp8));
    CUDA_CHECK(cudaMalloc(&d_C, size_C_fp16));

    // Pre-load all data before timing
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A_fp8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B_fp8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C_fp16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initialize cuBLASLt
    cublasLtHandle_t ltHandle;
    CUBLASLT_CHECK(cublasLtCreate(&ltHandle));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Create operation descriptor for FP8 GEMM
    cublasLtMatmulDesc_t matmulDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // Set transpose options (A and B are not transposed)
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Create matrix layout descriptors
    // Note: cuBLASLt uses CUDA_R_8F_E4M3 for FP8 E4M3 format
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8F_E4M3, K, M, K));  // A^T dimensions for col-major
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8F_E4M3, N, K, N));  // B dimensions
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16F, N, M, N));       // Output in FP16

    // Scaling factors for FP8
    float alpha = 1.0f;
    float beta = 0.0f;

    // Allocate workspace for cuBLASLt
    size_t workspaceSize = 1024 * 1024 * 4;  // 4MB workspace
    void* d_workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspaceSize));

    // Create preference for algorithm selection
    cublasLtMatmulPreference_t preference;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
                                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                         &workspaceSize,
                                                         sizeof(workspaceSize)));

    // Query and select best algorithm
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(ltHandle, matmulDesc, layoutA, layoutB, layoutC, layoutC,
                                                   preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        std::cerr << "No suitable algorithm found for FP8 GEMM" << std::endl;
        return 1;
    }

    // Warmup
    for (int batch = 0; batch < kBatchCount; ++batch) {
        const size_t offset_A = batch * elements_A;
        const size_t offset_B = batch * elements_B;
        const size_t offset_C = batch * elements_C;
        CUBLASLT_CHECK(cublasLtMatmul(ltHandle, matmulDesc,
                                       &alpha,
                                       d_B + offset_B, layoutB,
                                       d_A + offset_A, layoutA,
                                       &beta,
                                       d_C + offset_C, layoutC,
                                       d_C + offset_C, layoutC,
                                       &heuristicResult.algo,
                                       d_workspace, workspaceSize,
                                       stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Timed section: Kernel execution only
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int iter = 0; iter < kIterations; ++iter) {
        for (int batch = 0; batch < kBatchCount; ++batch) {
            const size_t offset_A = batch * elements_A;
            const size_t offset_B = batch * elements_B;
            const size_t offset_C = batch * elements_C;
            CUBLASLT_CHECK(cublasLtMatmul(ltHandle, matmulDesc,
                                           &alpha,
                                           d_B + offset_B, layoutB,
                                           d_A + offset_A, layoutA,
                                           &beta,
                                           d_C + offset_C, layoutC,
                                           d_C + offset_C, layoutC,
                                           &heuristicResult.algo,
                                           d_workspace, workspaceSize,
                                           stream));
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / static_cast<float>(kIterations * kBatchCount);
    
    // Calculate TFLOPS
    const double flops = 2.0 * M * N * K * kBatchCount * kIterations;
    const double tflops = flops / (total_ms * 1e9);
    
    std::cout << "cuBLASLt FP8 GEMM (tensor cores): " << avg_ms << " ms" << std::endl;
    std::cout << "Throughput: " << tflops << " TFLOPS" << std::endl;

    // Cleanup
    CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(matmulDesc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layoutA));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layoutB));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layoutC));
    CUBLASLT_CHECK(cublasLtDestroy(ltHandle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));

    return 0;
}






