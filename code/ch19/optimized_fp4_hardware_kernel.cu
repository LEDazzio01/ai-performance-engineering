/*
 * Optimized FP4 Hardware Kernel using cuBLASLt tensor cores.
 *
 * Uses native NVFP4 (CUDA_R_4F_E2M1) with VEC16_UE4M3 block scaling
 * for maximum tensor core throughput on Blackwell.
 *
 * Baseline uses manual FP4 packing without tensor cores.
 * This version demonstrates the massive speedup from proper tensor core usage.
 */

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CUBLASLT_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLASLt error " << __FILE__ << ":" << __LINE__ << " - " << status << std::endl; \
            exit(1); \
        } \
    } while(0)

constexpr int FP4_BLOCK_SIZE = 16;

// Quantize float to NVFP4 with block scaling
void quantize_to_nvfp4(const float* input, uint8_t* output_packed, 
                       __nv_fp8_e4m3* scales, int rows, int cols) {
    const int packed_cols = cols / 2;
    const int num_scale_cols = cols / FP4_BLOCK_SIZE;
    
    for (int r = 0; r < rows; ++r) {
        for (int block = 0; block < num_scale_cols; ++block) {
            const int block_start = block * FP4_BLOCK_SIZE;
            
            float max_abs = 0.0f;
            for (int i = 0; i < FP4_BLOCK_SIZE; ++i) {
                max_abs = std::max(max_abs, std::abs(input[r * cols + block_start + i]));
            }
            
            float scale = (max_abs > 0.0f) ? max_abs / 6.0f : 1.0f;
            scales[r * num_scale_cols + block] = __nv_fp8_e4m3(scale);
            
            for (int i = 0; i < FP4_BLOCK_SIZE; i += 2) {
                float v0 = input[r * cols + block_start + i];
                float v1 = input[r * cols + block_start + i + 1];
                
                __nv_fp4_storage_t fp4_0 = __nv_cvt_float_to_fp4(v0 / scale, __NV_E2M1, cudaRoundNearest);
                __nv_fp4_storage_t fp4_1 = __nv_cvt_float_to_fp4(v1 / scale, __NV_E2M1, cudaRoundNearest);
                
                int packed_idx = r * packed_cols + (block_start + i) / 2;
                output_packed[packed_idx] = ((fp4_1 & 0x0F) << 4) | (fp4_0 & 0x0F);
            }
        }
    }
}

int main() {
    std::cout << "=== Optimized FP4 Hardware Kernel (cuBLASLt Tensor Cores) ===" << std::endl;
    
    // Check GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (SM" << prop.major << "." << prop.minor << ")" << std::endl;
    
    // Matrix dimensions (aligned for tensor cores)
    const int M = 1024, N = 1024, K = 1024;
    
    // FP4 packed sizes
    const size_t packed_K = K / 2;
    const size_t packed_N = N / 2;
    const size_t elements_A_packed = M * packed_K;
    const size_t elements_B_packed = K * packed_N;
    const size_t elements_C = M * N;
    const size_t num_scales_A = M * (K / FP4_BLOCK_SIZE);
    const size_t num_scales_B = K * (N / FP4_BLOCK_SIZE);

    // Host allocation
    std::vector<float> h_A_fp32(M * K);
    std::vector<float> h_B_fp32(K * N);
    std::vector<uint8_t> h_A_packed(elements_A_packed);
    std::vector<uint8_t> h_B_packed(elements_B_packed);
    std::vector<__nv_fp8_e4m3> h_A_scales(num_scales_A);
    std::vector<__nv_fp8_e4m3> h_B_scales(num_scales_B);
    std::vector<__half> h_C(elements_C);

    // Initialize with random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& v : h_A_fp32) v = dis(gen);
    for (auto& v : h_B_fp32) v = dis(gen);

    // Quantize to NVFP4
    quantize_to_nvfp4(h_A_fp32.data(), h_A_packed.data(), h_A_scales.data(), M, K);
    quantize_to_nvfp4(h_B_fp32.data(), h_B_packed.data(), h_B_scales.data(), K, N);

    // Device allocation
    uint8_t *d_A = nullptr, *d_B = nullptr;
    __nv_fp8_e4m3 *d_A_scales = nullptr, *d_B_scales = nullptr;
    __half *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, elements_A_packed));
    CUDA_CHECK(cudaMalloc(&d_B, elements_B_packed));
    CUDA_CHECK(cudaMalloc(&d_A_scales, num_scales_A * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_B_scales, num_scales_B * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_C, elements_C * sizeof(__half)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A_packed.data(), elements_A_packed, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_packed.data(), elements_B_packed, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_scales, h_A_scales.data(), num_scales_A * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_scales, h_B_scales.data(), num_scales_B * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initialize cuBLASLt
    cublasLtHandle_t ltHandle;
    CUBLASLT_CHECK(cublasLtCreate(&ltHandle));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Create matmul descriptor
    cublasLtMatmulDesc_t matmulDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Set scale mode
    cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));

    void* d_A_scales_ptr = d_A_scales;
    void* d_B_scales_ptr = d_B_scales;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_A_scales_ptr, sizeof(d_A_scales_ptr)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_B_scales_ptr, sizeof(d_B_scales_ptr)));

    // Matrix layouts
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_4F_E2M1, M, K, M));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_4F_E2M1, K, N, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16F, M, N, M));

    float alpha = 1.0f, beta = 0.0f;

    // Workspace
    size_t workspaceSize = 1024 * 1024 * 32;
    void* d_workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspaceSize));

    // Algorithm selection
    cublasLtMatmulPreference_t preference;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                         &workspaceSize, sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    cublasStatus_t heuristicStatus = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, layoutA, layoutB, layoutC, layoutC,
        preference, 1, &heuristicResult, &returnedResults);
    
    if (heuristicStatus != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
        std::cerr << "No NVFP4 algorithm found, trying without block scaling..." << std::endl;
        cublasLtMatmulMatrixScale_t noScale = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &noScale, sizeof(noScale));
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &noScale, sizeof(noScale));
        heuristicStatus = cublasLtMatmulAlgoGetHeuristic(ltHandle, matmulDesc, layoutA, layoutB, layoutC, layoutC,
                                                          preference, 1, &heuristicResult, &returnedResults);
        if (returnedResults == 0) {
            std::cerr << "FP4 not supported on this system" << std::endl;
            return 1;
        }
    }

    // Warmup
    for (int i = 0; i < 5; ++i) {
        CUBLASLT_CHECK(cublasLtMatmul(ltHandle, matmulDesc, &alpha,
                                       d_A, layoutA, d_B, layoutB, &beta,
                                       d_C, layoutC, d_C, layoutC,
                                       &heuristicResult.algo, d_workspace, workspaceSize, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 100;
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) {
        CUBLASLT_CHECK(cublasLtMatmul(ltHandle, matmulDesc, &alpha,
                                       d_A, layoutA, d_B, layoutB, &beta,
                                       d_C, layoutC, d_C, layoutC,
                                       &heuristicResult.algo, d_workspace, workspaceSize, stream));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / iterations;

    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_ms * 1e9);

    std::cout << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "  Matrix: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "  Time: " << avg_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << tflops << " TFLOPS" << std::endl;
    std::cout << std::endl;
    std::cout << "Compare with baseline_fp4_hardware_kernel for speedup!" << std::endl;

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
    CUDA_CHECK(cudaFree(d_A_scales));
    CUDA_CHECK(cudaFree(d_B_scales));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
