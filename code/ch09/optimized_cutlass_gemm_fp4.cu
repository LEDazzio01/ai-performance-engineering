// optimized_cutlass_gemm_fp4.cu -- Native NVFP4 GEMM using cuBLASLt tensor cores
//
// This optimized version uses cuBLASLt with native NVFP4 (E2M1) data type and
// 16-element block scaling (VEC16_UE4M3) for maximum throughput on Blackwell.
//
// BOOK REFERENCE (Ch9/Ch19): NVFP4 tensor cores on Blackwell provide 3-5x
// throughput over FP16 using 4-bit precision with per-block scaling.
//
// REQUIRES: CUDA 12.9+, Blackwell GPU (SM 10.0+)

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <cmath>
#include <vector>

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

// Block size for NVFP4 scaling (16 elements per scale factor)
constexpr int FP4_BLOCK_SIZE = 16;

// Quantize float to NVFP4 with block scaling
// Each 16-element block shares one UE4M3 scale factor
void quantize_to_nvfp4(const float* input, uint8_t* output_packed, 
                       __nv_fp8_e4m3* scales,
                       int rows, int cols) {
    // FP4 is packed: 2 values per byte
    const int packed_cols = cols / 2;
    const int num_scale_cols = cols / FP4_BLOCK_SIZE;
    
    for (int r = 0; r < rows; ++r) {
        for (int block = 0; block < num_scale_cols; ++block) {
            const int block_start = block * FP4_BLOCK_SIZE;
            
            // Find max absolute value in this block
            float max_abs = 0.0f;
            for (int i = 0; i < FP4_BLOCK_SIZE; ++i) {
                max_abs = std::max(max_abs, std::abs(input[r * cols + block_start + i]));
            }
            
            // Scale factor: map max_abs to FP4 range [-6, 6]
            float scale = (max_abs > 0.0f) ? max_abs / 6.0f : 1.0f;
            
            // Store scale as UE4M3 (FP8 E4M3, used unsigned)
            scales[r * num_scale_cols + block] = __nv_fp8_e4m3(scale);
            
            // Quantize values in this block to FP4 (packed 2 per byte)
            for (int i = 0; i < FP4_BLOCK_SIZE; i += 2) {
                float v0 = input[r * cols + block_start + i];
                float v1 = input[r * cols + block_start + i + 1];
                
                // Convert to FP4 using CUDA conversion functions
                __nv_fp4_storage_t fp4_0 = __nv_cvt_float_to_fp4(v0 / scale, __NV_E2M1, cudaRoundNearest);
                __nv_fp4_storage_t fp4_1 = __nv_cvt_float_to_fp4(v1 / scale, __NV_E2M1, cudaRoundNearest);
                
                // Pack two FP4 values into one byte (low nibble = v0, high nibble = v1)
                int packed_idx = r * packed_cols + (block_start + i) / 2;
                output_packed[packed_idx] = ((fp4_1 & 0x0F) << 4) | (fp4_0 & 0x0F);
            }
        }
    }
}

int main() {
    // Check GPU architecture for NVFP4 support
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Running on " << prop.name << " (SM" << prop.major << "." << prop.minor << ")" << std::endl;
    
    if (prop.major < 10) {
        std::cerr << "ERROR: NVFP4 requires Blackwell (SM 10.0+), detected SM" 
                  << prop.major << "." << prop.minor << std::endl;
        return 1;
    }

    // Matrix dimensions - must be aligned to 16 for FP4 block scaling
    // Using dimensions that are multiples of 16 and 128 for tensor core alignment
    constexpr int M = 4096;  // Rows of A and C
    constexpr int N = 4096;  // Cols of B and C  
    constexpr int K = 4096;  // Cols of A, Rows of B
    constexpr int kIterations = 10;
    constexpr int kBatchCount = 8;
    
    // Verify alignment
    static_assert(M % FP4_BLOCK_SIZE == 0, "M must be multiple of 16");
    static_assert(N % FP4_BLOCK_SIZE == 0, "N must be multiple of 16");
    static_assert(K % FP4_BLOCK_SIZE == 0, "K must be multiple of 16");

    // FP4 sizes: packed (2 values per byte)
    const size_t packed_K = K / 2;   // A is MxK, packed along K
    const size_t packed_N = N / 2;   // B is KxN, packed along N
    const size_t elements_A_packed = static_cast<size_t>(M) * packed_K;
    const size_t elements_B_packed = static_cast<size_t>(K) * packed_N;
    const size_t elements_C = static_cast<size_t>(M) * N;
    
    // Scale tensor sizes: one scale per 16 elements along the packed dimension
    const size_t num_scales_per_row_A = K / FP4_BLOCK_SIZE;
    const size_t num_scales_per_row_B = N / FP4_BLOCK_SIZE;
    const size_t num_scales_A = M * num_scales_per_row_A;
    const size_t num_scales_B = K * num_scales_per_row_B;
    
    std::cout << "Matrix dimensions: M=" << M << " N=" << N << " K=" << K << std::endl;
    std::cout << "FP4 packed sizes: A=" << elements_A_packed << " B=" << elements_B_packed << std::endl;
    std::cout << "Scale counts: A=" << num_scales_A << " B=" << num_scales_B << std::endl;

    // Host allocation
    std::vector<float> h_A_fp32(M * K * kBatchCount);
    std::vector<float> h_B_fp32(K * N * kBatchCount);
    std::vector<uint8_t> h_A_packed(elements_A_packed * kBatchCount);
    std::vector<uint8_t> h_B_packed(elements_B_packed * kBatchCount);
    std::vector<__nv_fp8_e4m3> h_A_scales(num_scales_A * kBatchCount);
    std::vector<__nv_fp8_e4m3> h_B_scales(num_scales_B * kBatchCount);
    std::vector<__half> h_C(elements_C * kBatchCount);

    // Initialize with random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& v : h_A_fp32) v = dis(gen);
    for (auto& v : h_B_fp32) v = dis(gen);
    
    // Quantize to NVFP4 with block scaling
    std::cout << "Quantizing matrices to NVFP4..." << std::endl;
    for (int batch = 0; batch < kBatchCount; ++batch) {
        quantize_to_nvfp4(h_A_fp32.data() + batch * M * K,
                          h_A_packed.data() + batch * elements_A_packed,
                          h_A_scales.data() + batch * num_scales_A,
                          M, K);
        quantize_to_nvfp4(h_B_fp32.data() + batch * K * N,
                          h_B_packed.data() + batch * elements_B_packed,
                          h_B_scales.data() + batch * num_scales_B,
                          K, N);
    }
    
    std::fill(h_C.begin(), h_C.end(), __float2half(0.0f));

    // Device allocation
    uint8_t *d_A = nullptr, *d_B = nullptr;
    __nv_fp8_e4m3 *d_A_scales = nullptr, *d_B_scales = nullptr;
    __half *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, elements_A_packed * kBatchCount));
    CUDA_CHECK(cudaMalloc(&d_B, elements_B_packed * kBatchCount));
    CUDA_CHECK(cudaMalloc(&d_A_scales, num_scales_A * kBatchCount * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_B_scales, num_scales_B * kBatchCount * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_C, elements_C * kBatchCount * sizeof(__half)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A_packed.data(), elements_A_packed * kBatchCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_packed.data(), elements_B_packed * kBatchCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_scales, h_A_scales.data(), num_scales_A * kBatchCount * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_scales, h_B_scales.data(), num_scales_B * kBatchCount * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), elements_C * kBatchCount * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initialize cuBLASLt
    cublasLtHandle_t ltHandle;
    CUBLASLT_CHECK(cublasLtCreate(&ltHandle));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Create matmul descriptor for NVFP4 with FP32 compute
    cublasLtMatmulDesc_t matmulDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t transa = CUBLAS_OP_T;  // A is transposed for column-major
    cublasOperation_t transb = CUBLAS_OP_N;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Set scale mode to VEC16_UE4M3 (16-element blocks with UE4M3 scale factors)
    cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));

    // Set scale pointers
    void* d_A_scales_ptr = d_A_scales;
    void* d_B_scales_ptr = d_B_scales;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_A_scales_ptr, sizeof(d_A_scales_ptr)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_B_scales_ptr, sizeof(d_B_scales_ptr)));

    // Matrix layouts for NVFP4 (CUDA_R_4F_E2M1)
    // Note: Leading dimensions are in terms of elements, not bytes
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_4F_E2M1, M, K, M));  // A^T: KxM with ld=M
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_4F_E2M1, K, N, K));  // B: KxN with ld=K
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16F, M, N, M));       // C: MxN with ld=M

    float alpha = 1.0f;
    float beta = 0.0f;

    // Workspace allocation
    size_t workspaceSize = 1024 * 1024 * 64;  // 64MB workspace for FP4
    void* d_workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspaceSize));

    // Algorithm selection
    cublasLtMatmulPreference_t preference;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
                                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                         &workspaceSize,
                                                         sizeof(workspaceSize)));

    std::cout << "Querying cuBLASLt for NVFP4 algorithm..." << std::endl;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    cublasStatus_t heuristicStatus = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, layoutA, layoutB, layoutC, layoutC,
        preference, 1, &heuristicResult, &returnedResults);
    
    if (heuristicStatus != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
        std::cerr << "No suitable algorithm found for NVFP4 GEMM (status=" << heuristicStatus << ")" << std::endl;
        std::cerr << "This may indicate:" << std::endl;
        std::cerr << "  - CUDA driver version doesn't support FP4" << std::endl;
        std::cerr << "  - Matrix dimensions not supported" << std::endl;
        std::cerr << "  - cuBLASLt version doesn't have FP4 algorithms" << std::endl;
        
        // Try without scale modes as fallback diagnostic
        std::cout << "\nAttempting FP4 without block scaling for diagnostics..." << std::endl;
        cublasLtMatmulMatrixScale_t noScale = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &noScale, sizeof(noScale));
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &noScale, sizeof(noScale));
        
        heuristicStatus = cublasLtMatmulAlgoGetHeuristic(
            ltHandle, matmulDesc, layoutA, layoutB, layoutC, layoutC,
            preference, 1, &heuristicResult, &returnedResults);
        
        if (heuristicStatus == CUBLAS_STATUS_SUCCESS && returnedResults > 0) {
            std::cout << "FP4 works without block scaling - driver may not support VEC16_UE4M3" << std::endl;
        } else {
            std::cerr << "FP4 not supported at all on this system" << std::endl;
            return 1;
        }
    }

    std::cout << "NVFP4 GEMM algorithm found, running benchmark..." << std::endl;

    // Warmup
    for (int batch = 0; batch < kBatchCount; ++batch) {
        const size_t offset_A = batch * elements_A_packed;
        const size_t offset_B = batch * elements_B_packed;
        const size_t offset_C = batch * elements_C;
        CUBLASLT_CHECK(cublasLtMatmul(ltHandle, matmulDesc,
                                       &alpha,
                                       d_A + offset_A, layoutA,
                                       d_B + offset_B, layoutB,
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

    // Timed section
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int iter = 0; iter < kIterations; ++iter) {
        for (int batch = 0; batch < kBatchCount; ++batch) {
            const size_t offset_A = batch * elements_A_packed;
            const size_t offset_B = batch * elements_B_packed;
            const size_t offset_C = batch * elements_C;
            CUBLASLT_CHECK(cublasLtMatmul(ltHandle, matmulDesc,
                                           &alpha,
                                           d_A + offset_A, layoutA,
                                           d_B + offset_B, layoutB,
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
    
    std::cout << "cuBLASLt NVFP4 GEMM (tensor cores): " << avg_ms << " ms" << std::endl;
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
    CUDA_CHECK(cudaFree(d_A_scales));
    CUDA_CHECK(cudaFree(d_B_scales));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
