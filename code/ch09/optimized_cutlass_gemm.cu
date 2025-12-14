// optimized_cutlass_gemm.cu -- Optimized GEMM using cuBLASLt tensor cores.

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <random>

#include "../core/common/headers/cuda_verify.cuh"

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t status = (call);                                                 \
    if (status != cudaSuccess) {                                                 \
      std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " "           \
                << cudaGetErrorString(status) << std::endl;                      \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#define CUBLAS_CHECK(call)                                                       \
  do {                                                                           \
    cublasStatus_t status = (call);                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "cuBLAS error " << __FILE__ << ":" << __LINE__ << " code "    \
                << status << std::endl;                                          \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#define CUBLASLT_CHECK(call)                                                     \
  do {                                                                           \
    cublasStatus_t status = (call);                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "cuBLASLt error " << __FILE__ << ":" << __LINE__ << " code "  \
                << status << std::endl;                                          \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

int main() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;
    constexpr int batch_count = 32;
    constexpr int iterations = 5;
    constexpr size_t workspace_bytes = 64ull * 1024ull * 1024ull;

    const size_t elems_A = static_cast<size_t>(M) * K;
    const size_t elems_B = static_cast<size_t>(K) * N;
    const size_t elems_C = static_cast<size_t>(M) * N;
    const size_t size_A = elems_A * sizeof(float) * batch_count;
    const size_t size_B = elems_B * sizeof(float) * batch_count;
    const size_t size_C = elems_C * sizeof(float) * batch_count;

    float* h_A = nullptr;
    float* h_B = nullptr;
    float* h_C = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_A, size_A));
    CUDA_CHECK(cudaMallocHost(&h_B, size_B));
    CUDA_CHECK(cudaMallocHost(&h_C, size_C));

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < elems_A * batch_count; ++i) {
        h_A[i] = dis(gen);
    }
    for (size_t i = 0; i < elems_B * batch_count; ++i) {
        h_B[i] = dis(gen);
    }
    std::fill(h_C, h_C + (elems_C * batch_count), 0.0f);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

    cublasLtHandle_t ltHandle;
    CUBLASLT_CHECK(cublasLtCreate(&ltHandle));
    cublasHandle_t fallbackHandle;
    CUBLAS_CHECK(cublasCreate(&fallbackHandle));
    CUBLAS_CHECK(cublasSetMathMode(fallbackHandle, CUBLAS_TF32_TENSOR_OP_MATH));

    cublasLtMatmulDesc_t operationDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    cublasOperation_t trans = CUBLAS_OP_N;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, M, K, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, K, N, N));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, N));
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    const long long strideA = static_cast<long long>(elems_A);
    const long long strideB = static_cast<long long>(elems_B);
    const long long strideC = static_cast<long long>(elems_C);
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

    cublasLtMatmulPreference_t preference;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes,
        sizeof(workspace_bytes)));

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returnedResults = 0;
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        ltHandle,
        operationDesc,
        Adesc,
        Bdesc,
        Cdesc,
        Cdesc,
        preference,
        1,
        &heuristic,
        &returnedResults));
    
#if defined(CUBLASLT_ALGO_CAP_PROGRAMMATIC_DEPENDENT_LAUNCH)
    // CUDA 13 + Blackwell: Check for PDL (Programmatic Dependent Launch) support
    // PDL reduces kernel launch overhead for pipelined workloads
    int pdl_supported = 0;
    size_t pdl_size = sizeof(pdl_supported);
    if (returnedResults > 0) {
        cublasStatus_t pdl_status = cublasLtMatmulAlgoGetAttribute(
            ltHandle,
            &heuristic.algo,
            CUBLASLT_ALGO_CAP_PROGRAMMATIC_DEPENDENT_LAUNCH,
            &pdl_supported,
            sizeof(pdl_supported),
            &pdl_size);
        if (pdl_status == CUBLAS_STATUS_SUCCESS && pdl_supported) {
            std::cout << "PDL (Programmatic Dependent Launch) supported: YES" << std::endl;
            // PDL is automatically enabled when supported - no additional API calls needed
        }
    }
#endif

    void* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        if (returnedResults > 0) {
            CUBLASLT_CHECK(cublasLtMatmul(
                ltHandle,
                operationDesc,
                &alpha,
                d_A,
                Adesc,
                d_B,
                Bdesc,
                &beta,
                d_C,
                Cdesc,
                d_C,
                Cdesc,
                &heuristic.algo,
                workspace,
                workspace_bytes,
                0));
        } else {
            CUBLAS_CHECK(cublasSgemmStridedBatched(
                fallbackHandle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                M,
                K,
                &alpha,
                d_B,
                N,
                strideB,
                d_A,
                K,
                strideA,
                &beta,
                d_C,
                N,
                strideC,
                batch_count));
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / (iterations * batch_count);
    std::cout << "cuBLASLt batched GEMM (optimized): " << avg_ms << " ms" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    std::cout << "Checksum sample: " << h_C[0] << std::endl;

#ifdef VERIFY
    double checksum = 0.0;
    for (size_t i = 0; i < elems_C; ++i) {
        checksum += std::abs(static_cast<double>(h_C[i]));
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(workspace));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(operationDesc));
    CUBLASLT_CHECK(cublasLtDestroy(ltHandle));
    CUBLAS_CHECK(cublasDestroy(fallbackHandle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));

    return 0;
}
